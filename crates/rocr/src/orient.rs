//! PP-LCNet image classifiers used for orientation tasks:
//!
//! - **text-line orientation** (`PP-LCNet_x0_25_textline_ori`): 2 classes,
//!   0°/180°, input 160×80, tells whether a text-line crop is upside down.
//! - **document orientation** (`PP-LCNet_x1_0_doc_ori`): 4 classes,
//!   0°/90°/180°/270°, input 224×224, tells how a page is rotated.
//!
//! Both share the classic PP-LCNet architecture: a stem conv + depthwise-
//! separable blocks (DW k×k + optional SE + PW 1×1) with hardswish
//! activations, then a global average pool, 1×1 expand conv, and a linear head.
//! Weight layout is the transformers `PPLCNet` convention, verified against
//! the PaddleOCR transformers engine (see `scripts/verify_textline_ori.py` and
//! `scripts/verify_doc_ori.py`).

use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::{BatchNorm, ModuleT};
use image::DynamicImage;

use crate::common::preprocess::{doc_preprocess, image_to_tensor, DET_NORMALIZE};
use crate::error::Error;
use crate::model_loader;
use crate::rec::backbone::conv2d_2d;
use crate::rec::weights::W;

const BN_EPS: f64 = 1e-5;

/// Target input size of the textline orientation model (W × H).
pub const TEXTLINE_WIDTH: usize = 160;
pub const TEXTLINE_HEIGHT: usize = 80;

/// Hardswish activation: `x * clamp(x/6 + 0.5, 0, 1)` — matches torch
/// `nn.Hardswish`.
fn hardswish(x: &Tensor) -> Result<Tensor, Error> {
    let g = x.affine(1.0 / 6.0, 0.5)?.clamp(0f32, 1f32)?;
    Ok((x * g)?)
}

/// Convolution + batch norm + hardswish (torch `PPLCNetConvLayer`).
struct ConvBn {
    conv_w: Tensor,
    bn: BatchNorm,
}

impl ConvBn {
    fn new(w: &W, prefix: &str) -> Result<Self, Error> {
        let conv_w = w.get(&format!("{prefix}.convolution.weight"))?;
        let c_out = conv_w.dims()[0];
        let bn = BatchNorm::new(
            c_out,
            w.get(&format!("{prefix}.normalization.running_mean"))?,
            w.get(&format!("{prefix}.normalization.running_var"))?,
            w.get(&format!("{prefix}.normalization.weight"))?,
            w.get(&format!("{prefix}.normalization.bias"))?,
            BN_EPS,
        )?;
        Ok(Self { conv_w, bn })
    }

    fn forward(
        &self,
        x: &Tensor,
        pad: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> Result<Tensor, Error> {
        let y = conv2d_2d(x, &self.conv_w, None, pad, stride, groups)?;
        let y = self.bn.forward_t(&y, false)?;
        hardswish(&y)
    }
}

/// Squeeze-and-excitation channel attention: global avg pool → conv(relu) →
/// conv(hardsigmoid) → rescale. The hardsigmoid is torch's `nn.Hardsigmoid`
/// (`clamp(x/6 + 0.5, 0, 1)`), which is exactly `candle_nn::ops::hard_sigmoid`.
struct Se {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
}

impl Se {
    fn new(w: &W, prefix: &str) -> Result<Self, Error> {
        Ok(Self {
            conv1_w: w.get(&format!("{prefix}.convolutions.0.weight"))?,
            conv1_b: w.get(&format!("{prefix}.convolutions.0.bias"))?,
            conv2_w: w.get(&format!("{prefix}.convolutions.2.weight"))?,
            conv2_b: w.get(&format!("{prefix}.convolutions.2.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let s = x.mean_keepdim([2, 3])?;
        let s = conv2d_2d(&s, &self.conv1_w, Some(&self.conv1_b), (0, 0), (1, 1), 1)?.relu()?;
        let s = conv2d_2d(&s, &self.conv2_w, Some(&self.conv2_b), (0, 0), (1, 1), 1)?;
        let s = candle_nn::ops::hard_sigmoid(&s)?;
        Ok(x.broadcast_mul(&s)?)
    }
}

/// A depthwise-separable layer: DW k×k (groups=in) + optional SE + PW 1×1.
struct DepthwiseSeparable {
    c_in: usize,
    k: usize,
    stride: (usize, usize),
    dw: ConvBn,
    se: Option<Se>,
    pw: ConvBn,
}

impl DepthwiseSeparable {
    fn new(w: &W, prefix: &str, k: usize, stride: (usize, usize)) -> Result<Self, Error> {
        let dw = ConvBn::new(w, &format!("{prefix}.depthwise_convolution"))?;
        let c_in = dw.conv_w.dims()[0];
        let se_prefix = format!("{prefix}.squeeze_excitation_module");
        let se = if w.has(&format!("{se_prefix}.convolutions.0.weight")) {
            Some(Se::new(w, &se_prefix)?)
        } else {
            None
        };
        let pw = ConvBn::new(w, &format!("{prefix}.pointwise_convolution"))?;
        Ok(Self {
            c_in,
            k,
            stride,
            dw,
            se,
            pw,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let pad = (self.k / 2, self.k / 2);
        let y = self.dw.forward(x, pad, self.stride, self.c_in)?;
        let y = match &self.se {
            Some(se) => se.forward(&y)?,
            None => y,
        };
        self.pw.forward(&y, (0, 0), (1, 1), 1)
    }
}

/// The default PP-LCNet stage/block layout (transformers `PPLCNetConfig`),
/// used when `config.json` omits `block_configs` (e.g. the x1_0 classifier).
/// Each entry is `(kernel_size, stride)`.
fn default_block_configs() -> Vec<Vec<(usize, (usize, usize))>> {
    vec![
        vec![(3, (1, 1))],
        vec![(3, (2, 2)), (3, (1, 1))],
        vec![(3, (2, 2)), (3, (1, 1))],
        vec![
            (3, (2, 2)),
            (5, (1, 1)),
            (5, (1, 1)),
            (5, (1, 1)),
            (5, (1, 1)),
            (5, (1, 1)),
        ],
        vec![(5, (2, 2)), (5, (1, 1))],
    ]
}

/// Parsed `config.json` of a PP-LCNet classifier.
struct PpLcNetConfig {
    hidden_dropout_prob: f32,
    /// Per stage, per block: `(kernel_size, (stride_h, stride_w))`.
    stages: Vec<Vec<(usize, (usize, usize))>>,
}

impl PpLcNetConfig {
    fn from_json(v: &serde_json::Value) -> Result<Self, Error> {
        let stride_of = |s: &serde_json::Value| -> Result<(usize, usize), Error> {
            if let Some(u) = s.as_u64() {
                Ok((u as usize, u as usize))
            } else if let Some(a) = s.as_array() {
                Ok((
                    a[0].as_u64().unwrap_or(1) as usize,
                    a[1].as_u64().unwrap_or(1) as usize,
                ))
            } else {
                Err(Error::Config("unexpected stride value".into()))
            }
        };
        let stages = match v.get("block_configs").and_then(|x| x.as_array()) {
            Some(blocks) => {
                let mut stages = Vec::with_capacity(blocks.len());
                for stage in blocks {
                    let arr = stage
                        .as_array()
                        .ok_or_else(|| Error::Config("malformed block_configs stage".into()))?;
                    let mut layers = Vec::with_capacity(arr.len());
                    for blk in arr {
                        let k = blk[0].as_u64().unwrap_or(3) as usize;
                        let stride = stride_of(&blk[3])?;
                        layers.push((k, stride));
                    }
                    stages.push(layers);
                }
                stages
            }
            None => default_block_configs(),
        };
        Ok(Self {
            hidden_dropout_prob: v
                .get("hidden_dropout_prob")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0) as f32,
            stages,
        })
    }
}

/// A generic PP-LCNet image classifier (weights from a transformers-style
/// safetensors repository).
pub struct PpLcNet {
    stem: ConvBn,
    stages: Vec<Vec<DepthwiseSeparable>>,
    last_conv_w: Tensor,
    head_w: Tensor,
    head_b: Tensor,
    dropout: f32,
    device: Device,
}

impl PpLcNet {
    /// Load the model from its repository directory.
    pub fn new(repo: &Path, device: &Device) -> Result<Self, Error> {
        let config = model_loader::load_json(repo, model_loader::CONFIG_FILE)?;
        let cfg = PpLcNetConfig::from_json(&config)?;
        let w = W(&model_loader::load_tensors(repo, device)?);
        let stem = ConvBn::new(&w, "encoder.convolution")?;
        let mut stages = Vec::with_capacity(cfg.stages.len());
        for (si, stage) in cfg.stages.iter().enumerate() {
            let mut layers = Vec::with_capacity(stage.len());
            for (li, (k, stride)) in stage.iter().enumerate() {
                let prefix = format!("encoder.blocks.{si}.layers.{li}");
                layers.push(DepthwiseSeparable::new(&w, &prefix, *k, *stride)?);
            }
            stages.push(layers);
        }
        Ok(Self {
            stem,
            stages,
            last_conv_w: w.get("last_convolution.weight")?,
            head_w: w.get("head.weight")?,
            head_b: w.get("head.bias")?,
            dropout: cfg.hidden_dropout_prob,
            device: device.clone(),
        })
    }

    /// Run the model on a preprocessed `[1,3,H,W]` tensor, returning the class
    /// logits `[1,C]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let mut x = self.stem.forward(x, (1, 1), (2, 2), 1)?;
        for stage in &self.stages {
            for layer in stage {
                x = layer.forward(&x)?;
            }
        }
        // Global average pool → 1×1, then the 1×1 expand conv + hardswish.
        let x = x.mean_keepdim([2, 3])?;
        let x = conv2d_2d(&x, &self.last_conv_w, None, (0, 0), (1, 1), 1)?;
        let x = hardswish(&x)?;
        // The transformers implementation scales by `1 - dropout` at inference
        // time too, so reproduce that here for exact parity.
        let x = x.affine((1.0 - self.dropout).into(), 0.0)?;
        let (n, c, _, _) = x.dims4()?;
        let x = x.reshape((n, c))?;
        let out = x.matmul(&self.head_w.t()?)?;
        Ok(out.broadcast_add(&self.head_b)?)
    }
}

/// Classify a text-line crop: `true` when it is upside down (180°) and should
/// be rotated before recognition.
pub fn classify_textline(net: &PpLcNet, crop: &DynamicImage) -> Result<bool, Error> {
    let x = image_to_tensor(
        crop,
        TEXTLINE_WIDTH,
        TEXTLINE_HEIGHT,
        &DET_NORMALIZE,
        &net.device,
    )?;
    let logits = net.forward(&x)?;
    let v = logits.to_vec2::<f32>()?;
    Ok(v[0][1] > v[0][0])
}

/// Classify a full page's rotation. Returns 0/1/2/3 for 0°/90°/180°/270°.
pub fn classify_doc(net: &PpLcNet, img: &DynamicImage) -> Result<u32, Error> {
    let x = doc_preprocess(img, &net.device)?;
    let logits = net.forward(&x)?;
    let v = logits.to_vec2::<f32>()?;
    let class = v[0]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    Ok(class)
}
