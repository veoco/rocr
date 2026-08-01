//! Text detection: shared LCNetV4 backbone (detection mode) + RepLKFPN +
//! DB head, plus DB post-processing.
//!
//! The backbone is the same LCNetV4 used by recognition, but with symmetric
//! stride-2 downsampling and stage outputs fed to the FPN.

mod db_postprocess;
mod medium_neck;
pub mod repfpn;

use std::path::Path;

use candle_core::{Device, Tensor};
use image::DynamicImage;

use crate::common::preprocess::det_preprocess;
use crate::error::Error;
use crate::model_loader;
use crate::rec::backbone::Backbone;
use crate::rec::config::BackboneConfig;
use crate::rec::weights::W;

use self::db_postprocess::DbPostprocess;
use self::medium_neck::{MediumDetConfig, MediumDetNeck};
use self::repfpn::{DbHead, RepLkFpn};

/// Detection model configuration.
pub struct DetConfig {
    pub backbone: BackboneConfig,
}

impl DetConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, Error> {
        Ok(Self {
            backbone: BackboneConfig::from_json(&v["backbone_config"])?,
        })
    }
}

/// The detection neck — the small tier uses a RepLKFPN, the medium tier uses a
/// PAN with intra-class blocks.
enum DetNeck {
    Small(RepLkFpn),
    Medium(MediumDetNeck),
}

impl DetNeck {
    fn forward(&self, stages: &[Tensor]) -> Result<Tensor, Error> {
        match self {
            DetNeck::Small(n) => n.forward(stages),
            DetNeck::Medium(n) => n.forward(stages),
        }
    }

    fn head(&self) -> &DbHead {
        match self {
            DetNeck::Small(n) => n.head(),
            DetNeck::Medium(n) => n.head(),
        }
    }
}

/// The PP-OCRv6 text detection model.
pub struct DetModel {
    backbone: Backbone,
    neck: DetNeck,
    post: DbPostprocess,
    device: Device,
}

impl DetModel {
    /// Load the detection model from a model repository directory.
    pub fn new(repo: &Path, device: &Device) -> Result<Self, Error> {
        let config = model_loader::load_json(repo, model_loader::CONFIG_FILE)?;
        let cfg = DetConfig::from_json(&config)?;
        let w = W(&model_loader::load_tensors(repo, device)?);
        let backbone = Backbone::new(&w, &cfg.backbone)?;
        // The small and medium tiers use different neck structures; dispatch on
        // which weight group is present.
        let neck = if w.has("model.neck.intraclass_blocks.0.conv_reduce_channel.weight") {
            let mcfg = MediumDetConfig::from_json(&config)?;
            DetNeck::Medium(MediumDetNeck::new(&w, &mcfg)?)
        } else {
            DetNeck::Small(RepLkFpn::new(&w)?)
        };
        let post = DbPostprocess::default();
        Ok(Self {
            backbone,
            neck,
            post,
            device: device.clone(),
        })
    }

    /// Run the detection model, returning the sigmoid probability map
    /// `[B, 1, H, W]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let stages = self.backbone.forward_stages(x)?;
        let fpn = self.neck.forward(&stages)?;
        self.neck.head().forward(&fpn)
    }

    /// Detect text boxes in an image. Returns the four-corner polygons in the
    /// original image coordinate space.
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<Vec<[f32; 2]>>, Error> {
        let (x, nh, nw) = det_preprocess(img, &self.device)?;
        let prob = self.forward(&x)?;
        let (h0, w0) = (img.height() as f32, img.width() as f32);
        let boxes = self.post.run(&prob, &self.device)?;
        // scale boxes from the resized input space back to the original image
        let sx = w0 / nw as f32;
        let sy = h0 / nh as f32;
        Ok(boxes
            .into_iter()
            .map(|poly| poly.into_iter().map(|[x, y]| [x * sx, y * sy]).collect())
            .collect())
    }
}

#[cfg(test)]
mod debug_tests {
    use std::path::PathBuf;

    use super::*;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn load_ref(name: &str) -> candle_core::Tensor {
        let p = root()
            .join("dev-models/reference/det_inter")
            .join(format!("{}.npy", name.replace('.', "_")));
        candle_core::Tensor::read_npy(p).unwrap()
    }

    /// Structural sanity: the backbone stages should match the ONNX reference
    /// closely. The neck weights of the det ONNX differ from the safetensors
    /// (the ONNX was exported with BN folded into the weights), so only the
    /// backbone is numerically compared.
    #[test]
    fn backbone_matches_oracle() {
        let repo = root().join("dev-models/PP-OCRv6_small_det_safetensors");
        if !repo.exists() || !root().join("dev-models/reference/det_inter").exists() {
            eprintln!("skip: det repo or det_inter fixtures not present");
            return;
        }
        let device = Device::Cpu;
        let config = model_loader::load_json(&repo, model_loader::CONFIG_FILE).unwrap();
        let cfg = DetConfig::from_json(&config).unwrap();
        let w = W(&model_loader::load_tensors(&repo, &device).unwrap());
        let backbone = Backbone::new(&w, &cfg.backbone).unwrap();
        let x =
            candle_core::Tensor::read_npy(root().join("dev-models/reference/det_input_small.npy"))
                .unwrap();
        let stages = backbone.forward_stages(&x).unwrap();
        for (i, ref_name) in ["Add.33", "Add.65", "Add.121", "Add.153"]
            .iter()
            .enumerate()
        {
            let diff =
                crate::common::preprocess::max_abs_diff(&stages[i], &load_ref(ref_name)).unwrap();
            println!("stage {i} diff = {diff}");
            assert!(diff < 0.02, "stage {i} deviates: {diff}");
        }
    }

    /// Verify the DB head and post-process numerically: feeding the ONNX FPN
    /// concat must reproduce the ONNX output, and the post-process must recover
    /// the 6 text lines from the dense ONNX probability map.
    #[test]
    fn head_and_postprocess_are_correct() {
        let repo = root().join("dev-models/PP-OCRv6_small_det_safetensors");
        if !repo.exists() || !root().join("dev-models/reference/det_inter").exists() {
            eprintln!("skip: det repo or det_inter fixtures not present");
            return;
        }
        let device = Device::Cpu;
        let model = DetModel::new(&repo, &device).unwrap();
        let onnx_concat = candle_core::Tensor::read_npy(
            root().join("dev-models/reference/det_inter/Concat_3.npy"),
        )
        .unwrap();
        let prob = model.neck.head().forward(&onnx_concat).unwrap();
        let ref_final =
            candle_core::Tensor::read_npy(root().join("dev-models/reference/det_output_small.npy"))
                .unwrap();
        let diff = crate::common::preprocess::max_abs_diff(&prob, &ref_final).unwrap();
        println!("head-on-onnx-concat diff = {diff}");
        assert!(diff < 1e-3, "head deviates: {diff}");

        let onnx_prob =
            candle_core::Tensor::read_npy(root().join("dev-models/reference/det_output_small.npy"))
                .unwrap();
        let boxes = model.post.run(&onnx_prob, &device).unwrap();
        println!("postprocess on ONNX prob map -> {} boxes", boxes.len());
        assert_eq!(boxes.len(), 6, "expected 6 text boxes");
    }
}
