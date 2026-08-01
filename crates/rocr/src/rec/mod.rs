//! Text recognition: PPLCNetV4 backbone (recognition mode) + LightSVTR +
//! CTC head, with greedy CTC decoding.
//!
//! Implemented and validated against the official PP-OCRv6 ONNX export
//! (see `crates/rocr/tests/oracle.rs`).

pub(crate) mod backbone;
pub(crate) mod config;
mod ctc_decode;
pub(crate) mod lightsvtr;
pub(crate) mod weights;

use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::ModuleT;
use image::DynamicImage;

use crate::common::preprocess::rec_preprocess;
use crate::error::Error;
use crate::model_loader;

use self::backbone::Backbone;
use self::config::RecConfig;
use self::ctc_decode::ctc_greedy;
use self::lightsvtr::LightSvtr;
use self::weights::W;

const BN_EPS: f64 = 1e-5;

/// `y = x @ wᵀ + b` for a 3D `x` and a 2D weight `[N, K]` / bias `[N]`.
pub(crate) fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    let w = w.t()?.unsqueeze(0)?;
    Ok(x.matmul(&w)?.broadcast_add(b)?)
}

/// Hardswish activation: `x * clamp(x/6 + 0.5, 0, 1)`.
fn hardswish(x: &Tensor) -> Result<Tensor, Error> {
    let g = x.affine(1.0 / 6.0, 0.5)?.clamp(0f32, 1f32)?;
    Ok((x * g)?)
}

/// Tiny-tier recognition head: no LightSVTR neck; a 1D depthwise conv stack
/// followed by two fully-connected layers.
struct TinyRecHead {
    conv1_w: Tensor,
    norm1: candle_nn::BatchNorm,
    conv2_w: Tensor,
    norm2: candle_nn::BatchNorm,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

impl TinyRecHead {
    fn new(w: &W) -> Result<Self, Error> {
        let norm = |p: &str, c: usize| -> Result<candle_nn::BatchNorm, Error> {
            Ok(candle_nn::BatchNorm::new(
                c,
                w.get(&format!("{p}.running_mean"))?,
                w.get(&format!("{p}.running_var"))?,
                w.get(&format!("{p}.weight"))?,
                w.get(&format!("{p}.bias"))?,
                BN_EPS,
            )?)
        };
        let out = w.get("head.conv1.weight")?.dims()[0];
        Ok(Self {
            conv1_w: w.get("head.conv1.weight")?,
            norm1: norm("head.norm1", out)?,
            conv2_w: w.get("head.conv2.weight")?,
            norm2: norm("head.norm2", out)?,
            fc1_w: w.get("head.fc1.weight")?,
            fc1_b: w.get("head.fc1.bias")?,
            fc2_w: w.get("head.fc2.weight")?,
            fc2_b: w.get("head.fc2.bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // x: [B, C, 3, T] → avgpool → [B, C, 1, T/2]
        let x = x.avg_pool2d_with_stride((3, 2), (3, 2))?.squeeze(2)?; // [B, C, L]
        let (_, _, kh) = self.conv1_w.dims3()?;
        let groups = self.conv1_w.dims()[0];
        let x = x.conv1d(&self.conv1_w, kh / 2, 1, 1, groups)?;
        let x = hardswish(&self.norm1.forward_t(&x, false)?)?;
        let x = x.conv1d(&self.conv2_w, 0, 1, 1, 1)?;
        let x = hardswish(&self.norm2.forward_t(&x, false)?)?;
        let x = x.transpose(1, 2)?; // [B, L, C]
        let x = linear(&x, &self.fc1_w, &self.fc1_b)?;
        let x = linear(&x, &self.fc2_w, &self.fc2_b)?;
        Ok(candle_nn::ops::softmax_last_dim(&x)?)
    }
}

/// The recognition neck + head — either the standard LightSVTR + CTC head or
/// the tiny-tier head.
enum RecHead {
    Standard(LightSvtr, Tensor, Tensor),
    Tiny(TinyRecHead),
}

impl RecHead {
    fn forward(&self, backbone_out: &Tensor) -> Result<Tensor, Error> {
        match self {
            RecHead::Standard(neck, head_w, head_b) => {
                let y = neck.forward(backbone_out)?; // [B, T, H]
                let logits = linear(&y, head_w, head_b)?;
                Ok(candle_nn::ops::softmax_last_dim(&logits)?)
            }
            RecHead::Tiny(head) => head.forward(backbone_out),
        }
    }
}

/// The PP-OCRv6 text recognition model.
pub struct RecModel {
    backbone: Backbone,
    head: RecHead,
    chars: Vec<String>,
    device: Device,
}

impl RecModel {
    /// Load the recognition model from a model repository directory.
    pub fn new(repo: &Path, device: &Device) -> Result<Self, Error> {
        let config = model_loader::load_json(repo, model_loader::CONFIG_FILE)?;
        let cfg = RecConfig::from_json(&config)?;
        let pre = model_loader::load_json(repo, model_loader::PREPROCESSOR_CONFIG_FILE)?;
        let chars: Vec<String> = pre
            .get("character_list")
            .and_then(|c| c.as_array())
            .ok_or_else(|| Error::Config("missing character_list".into()))?
            .iter()
            .map(|c| c.as_str().unwrap_or("").to_string())
            .collect();
        let w = W(&model_loader::load_tensors(repo, device)?);
        let backbone = Backbone::new(&w, &cfg.backbone)?;
        let head = if w.has("head.conv1.weight") {
            RecHead::Tiny(TinyRecHead::new(&w)?)
        } else {
            let neck = LightSvtr::new(&w, &cfg)?;
            let head_w = w.get("head.head.weight")?;
            let head_b = w.get("head.head.bias")?;
            RecHead::Standard(neck, head_w, head_b)
        };
        Ok(Self {
            backbone,
            head,
            chars,
            device: device.clone(),
        })
    }

    /// Run the model, returning per-position class probabilities `[B, T, C]`
    /// (softmax applied, matching the official ONNX export).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = self.backbone.forward(x)?;
        self.head.forward(&y)
    }

    /// Recognize a single text-line image crop.
    pub fn recognize(&self, crop: &DynamicImage) -> Result<(String, f32), Error> {
        let (x, _w) = rec_preprocess(crop, &self.device)?;
        let probs = self.forward(&x)?; // [1, T, C]
        let decoded = ctc_greedy(&probs.narrow(0, 0, 1)?.squeeze(0)?)?;
        let text: String = decoded
            .ids
            .iter()
            .filter_map(|&i| self.chars.get(i).cloned())
            .collect();
        let conf = if decoded.confidences.is_empty() {
            0.0
        } else {
            decoded.confidences.iter().sum::<f32>() / decoded.confidences.len() as f32
        };
        Ok((text, conf))
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn load_ref(name: &str) -> Tensor {
        let path = root().join("dev-models/reference/rec_inter").join(name);
        Tensor::read_npy(path).unwrap()
    }

    #[test]
    fn compare_stem_and_backbone() {
        let repo = root().join("dev-models/PP-OCRv6_small_rec_safetensors");
        if !repo.exists() || !root().join("dev-models/reference/rec_inter").exists() {
            eprintln!("skip: rec repo or rec_inter fixtures not present");
            return;
        }
        let device = Device::Cpu;
        let config = model_loader::load_json(&repo, model_loader::CONFIG_FILE).unwrap();
        let cfg = RecConfig::from_json(&config).unwrap();
        let w = W(&model_loader::load_tensors(&repo, &device).unwrap());
        let backbone = Backbone::new(&w, &cfg.backbone).unwrap();
        let x = Tensor::read_npy(root().join("dev-models/reference/rec_input_small.npy")).unwrap();

        let stem = backbone.stem_forward(&x).unwrap();
        let diff =
            crate::common::preprocess::max_abs_diff(&stem, &load_ref("p2o_pd_op_relu_4_0.npy"))
                .unwrap();
        println!("stem diff = {diff}");
        assert!(diff < 1e-3, "stem deviates: {diff}");

        let full = backbone.forward(&x).unwrap();
        let diff =
            crate::common::preprocess::max_abs_diff(&full, &load_ref("p2o_pd_op_add_64_0.npy"))
                .unwrap();
        println!("backbone diff = {diff}");
        assert!(diff < 1e-3, "backbone deviates: {diff}");

        let neck = lightsvtr::LightSvtr::new(&w, &cfg).unwrap();
        let pooled = full.avg_pool2d_with_stride((3, 2), (3, 2)).unwrap();
        let diff =
            crate::common::preprocess::max_abs_diff(&pooled, &load_ref("p2o_pd_op_pool2d_1_0.npy"))
                .unwrap();
        println!("pool diff = {diff}");
        assert!(diff < 1e-3, "pool deviates: {diff}");

        let (a_branch, main_bt) = neck.debug_pre_svtr(&full).unwrap();
        let _ = a_branch;
        // compare b+c against add.65.0 [1,120,1,66]
        let ref_65 = load_ref("p2o_pd_op_add_65_0.npy");
        let ref_65_bt = ref_65.squeeze(2).unwrap().permute((0, 2, 1)).unwrap();
        let diff = crate::common::preprocess::max_abs_diff(&main_bt, &ref_65_bt).unwrap();
        println!("b+c diff = {diff}");
        assert!(diff < 1e-3, "b+c deviates: {diff}");

        // Compare each svtr step against ONNX intermediates.
        let steps = neck.debug_svtr(&full).unwrap();
        let refs = [
            ("main", "p2o_pd_op_transpose_0_0.npy"),
            ("ln0", "p2o_pd_op_layer_norm_0_0.npy"),
            ("after_attn0", "p2o_pd_op_add_68_0.npy"),
            ("ln1", "p2o_pd_op_layer_norm_1_0.npy"),
            ("after_mlp0", "p2o_pd_op_add_71_0.npy"),
            ("ln2", "p2o_pd_op_layer_norm_2_0.npy"),
            ("after_attn1", "p2o_pd_op_add_74_0.npy"),
            ("ln3", "p2o_pd_op_layer_norm_3_0.npy"),
            ("after_mlp1", "p2o_pd_op_add_77_0.npy"),
        ];
        for (i, (label, ref_name)) in refs.iter().enumerate() {
            let ref_t = load_ref(ref_name);
            let diff = crate::common::preprocess::max_abs_diff(&steps[i], &ref_t).unwrap();
            println!("svtr {label:<12} diff = {diff}");
            assert!(diff < 1e-2, "svtr {label} deviates: {diff}");
        }

        let neck_out = neck.forward(&full).unwrap(); // [B,T,H]
        let ref_add78 = load_ref("p2o_pd_op_add_78_0.npy");
        let ref_as_bt = ref_add78.squeeze(2).unwrap().permute((0, 2, 1)).unwrap();
        let diff = crate::common::preprocess::max_abs_diff(&neck_out, &ref_as_bt).unwrap();
        println!("neck final diff = {diff}");
        assert!(diff < 1e-2, "neck deviates: {diff}");
    }
}
