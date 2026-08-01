//! rocr — a pure Rust OCR library powered by PaddlePaddle PP-OCRv6 models.
//!
//! The library loads PP-OCRv6 model weights (manually downloaded by the user)
//! and runs the full OCR pipeline with the [candle] tensor framework, with no
//! C++ inference backend.
//!
//! ```no_run
//! use rocr::{Ocr, OcrConfig, ModelTier, DeviceKind};
//!
//! let ocr = Ocr::new(OcrConfig {
//!     model_tier: ModelTier::Small,
//!     device: DeviceKind::Cpu,
//!     model_dir: "path/to/models".into(),
//!     ..Default::default()
//! })?;
//! let img = image::open("image.png")?;
//! let results = ocr.recognize(&img)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod common;
pub mod det;
pub mod error;
pub mod model_loader;
pub mod orient;
pub mod pipeline;
pub mod rec;
pub mod unwarp;

pub use det::DetModel;
pub use orient::PpLcNet;
pub use rec::RecModel;

use std::path::PathBuf;

pub use error::Error;

/// Model size tier. All tiers share the same architecture primitives and
/// differ only through their `config.json` parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelTier {
    Tiny,
    #[default]
    Small,
    Medium,
}

impl ModelTier {
    /// The model family prefix used by the Hugging Face repositories.
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            ModelTier::Tiny => "tiny",
            ModelTier::Small => "small",
            ModelTier::Medium => "medium",
        }
    }
}

/// Inference device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceKind {
    #[default]
    Cpu,
    Cuda,
    Metal,
}

/// Configuration for an [`Ocr`] instance.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    pub model_tier: ModelTier,
    pub device: DeviceKind,
    /// Rotate the whole document to the correct orientation (0/90/180/270).
    /// Default off — requires an extra doc-orientation model.
    pub enable_doc_orientation: bool,
    /// Rectify curved document pages. Default off — requires an extra
    /// unwarping model.
    pub enable_doc_unwarping: bool,
    /// Classify each detected text line for 180° rotation. Default on.
    pub enable_textline_orientation: bool,
    /// Repository name of the text-line orientation model. Defaults to
    /// `PP-LCNet_x1_0_textline_ori_safetensors` (the PaddleOCR default); set to
    /// `Some("PP-LCNet_x0_25_textline_ori_safetensors")` for the lighter model.
    pub textline_ori_model_name: Option<String>,
    /// Directory containing the manually downloaded model repositories.
    pub model_dir: PathBuf,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            model_tier: ModelTier::default(),
            device: DeviceKind::default(),
            enable_doc_orientation: false,
            enable_doc_unwarping: false,
            enable_textline_orientation: true,
            textline_ori_model_name: None,
            model_dir: PathBuf::new(),
        }
    }
}

/// A single OCR result: recognized text, confidence and bounding polygon.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    /// Four corner points `[x, y]` in the original image coordinate space.
    pub polygon: Vec<[f32; 2]>,
}

/// Default repository name of the text-line orientation model (fixed across
/// tiers). The x1_0 model is what PaddleOCR itself uses by default; the lighter
/// x0_25 variant is available via `OcrConfig::textline_ori_model_name`.
const TEXTLINE_ORI_REPO: &str = "PP-LCNet_x1_0_textline_ori_safetensors";
/// Repository name of the document orientation model.
const DOC_ORI_REPO: &str = "PP-LCNet_x1_0_doc_ori_safetensors";
/// Repository name of the document unwarping model.
const UNWARP_REPO: &str = "UVDoc_safetensors";

/// The OCR engine. Owns the loaded detection / recognition models.
pub struct Ocr {
    det: DetModel,
    rec: RecModel,
    /// Text-line orientation classifier (0°/180°), applied per crop.
    textline_ori: Option<PpLcNet>,
    /// Document orientation classifier (0°/90°/180°/270°), applied to the page.
    doc_ori: Option<PpLcNet>,
    /// Document unwarping (UVDoc), applied to the whole page when enabled.
    unwarp: Option<crate::unwarp::UVDoc>,
}

impl Ocr {
    /// Build an [`Ocr`] engine, loading the detection and recognition models
    /// from `config.model_dir`.
    pub fn new(config: OcrConfig) -> Result<Self, Error> {
        configure_candle_threads();
        let device = device_from_kind(config.device)?;
        let tier = config.model_tier.as_str();
        let base = &config.model_dir;
        if !base.exists() {
            return Err(Error::ModelDirMissing(base.clone()));
        }
        let det_repo = model_loader::repo_dir(base, &format!("{tier}_det"));
        let rec_repo = model_loader::repo_dir(base, &format!("{tier}_rec"));
        let det = DetModel::new(&det_repo, &device)?;
        let rec = RecModel::new(&rec_repo, &device)?;
        // Text-line orientation classification (default on, mirroring
        // PaddleOCR). The model repositories have fixed names across tiers.
        let load_cls = |repo_name: &str, enabled: bool| -> Result<Option<PpLcNet>, Error> {
            if !enabled {
                return Ok(None);
            }
            let repo = base.join(repo_name);
            if !repo.exists() {
                return Err(Error::ModelFileMissing(repo.display().to_string()));
            }
            Ok(Some(PpLcNet::new(&repo, &device)?))
        };
        let textline_ori_repo = config
            .textline_ori_model_name
            .as_deref()
            .unwrap_or(TEXTLINE_ORI_REPO);
        let textline_ori = load_cls(textline_ori_repo, config.enable_textline_orientation)?;
        let doc_ori = load_cls(DOC_ORI_REPO, config.enable_doc_orientation)?;
        let unwarp = if config.enable_doc_unwarping {
            let repo = base.join(UNWARP_REPO);
            if !repo.exists() {
                return Err(Error::ModelFileMissing(repo.display().to_string()));
            }
            Some(crate::unwarp::UVDoc::new(&repo, &device)?)
        } else {
            None
        };
        Ok(Self {
            det,
            rec,
            textline_ori,
            doc_ori,
            unwarp,
        })
    }

    /// Run the full OCR pipeline on an image: (optional) document rotation
    /// correction → detection → per-box text-line orientation → recognition.
    pub fn recognize(&self, img: &image::DynamicImage) -> Result<Vec<OcrResult>, Error> {
        let mut page = img.clone();
        // Correct the page orientation if the doc-orientation module is on.
        if let Some(doc) = &self.doc_ori {
            match orient::classify_doc(doc, &page)? {
                // class k → the page was rotated `k×90°` counter-clockwise, so
                // rotate it clockwise by that amount to restore it.
                1 => page = page.rotate270(),
                2 => page = page.rotate180(),
                3 => page = page.rotate90(),
                _ => {}
            }
        }
        // Rectify curved pages before detection if unwarping is enabled.
        if let Some(unwarp) = &self.unwarp {
            page = unwarp.unwarp_image(&page)?;
        }
        let boxes = self.det.detect(&page)?;
        let mut results = Vec::new();
        for poly in boxes {
            let Some(crop) = crop_polygon(&page, &poly) else {
                continue;
            };
            // Classify the line orientation and rotate if upside down.
            let crop = match &self.textline_ori {
                Some(tl) if orient::classify_textline(tl, &crop)? => crop.rotate180(),
                _ => crop,
            };
            let (text, confidence) = self.rec.recognize(&crop)?;
            if text.is_empty() {
                continue;
            }
            results.push(OcrResult {
                text,
                confidence,
                polygon: poly,
            });
        }
        Ok(results)
    }

    /// Run text detection only.
    pub fn detect(&self, img: &image::DynamicImage) -> Result<Vec<Vec<[f32; 2]>>, Error> {
        self.det.detect(img)
    }

    /// Recognize a set of cropped text-line images.
    pub fn recognize_crops(
        &self,
        crops: &[image::DynamicImage],
    ) -> Result<Vec<(String, f32)>, Error> {
        let mut out = Vec::with_capacity(crops.len());
        for crop in crops {
            out.push(self.rec.recognize(crop)?);
        }
        Ok(out)
    }
}

/// Crop the axis-aligned bounding box of a polygon from an image.
pub(crate) fn crop_polygon(
    img: &image::DynamicImage,
    poly: &[[f32; 2]],
) -> Option<image::DynamicImage> {
    let mut x0 = f32::INFINITY;
    let mut y0 = f32::INFINITY;
    let mut x1 = f32::NEG_INFINITY;
    let mut y1 = f32::NEG_INFINITY;
    for p in poly {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let x0 = (x0.floor().max(0.0)) as u32;
    let y0 = (y0.floor().max(0.0)) as u32;
    let x1 = (x1.ceil().min(iw)) as u32;
    let y1 = (y1.ceil().min(ih)) as u32;
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(img.crop_imm(x0, y0, x1 - x0, y1 - y0))
}

/// Default CPU thread count for candle's internal thread pool.
///
/// candle builds its rayon thread pool lazily on the first tensor op, sized by
/// `num_cpus::get_physical()`. On machines with many cores this makes the small
/// convolution-heavy ops of PP-OCRv6 dominated by barrier sync: on a 16-physical-core
/// host a full-page OCR took ~59 s with the default pool vs ~4.7 s with
/// 6 threads. Since the pool is only configurable through the environment, we
/// default to a single thread — the most portable choice (performance across
/// thread counts is hardware-specific; users who want more should set
/// `RAYON_NUM_THREADS` or use the CLI `--threads`).
const DEFAULT_CANDLE_THREADS: usize = 1;

/// Clamp candle's internal rayon thread pool to a sane default.
///
/// Must run before any candle tensor op (candle's pool is a lazy `OnceLock`,
/// so the env var is only honored if set before the first op). This function
/// must be called at the top of `Ocr::new`. It is a no-op when the user has
/// already set `RAYON_NUM_THREADS` themselves.
fn configure_candle_threads() {
    // Nothing else touches this env var in rocr; callers can override via the
    // environment or the CLI. The candle pool is created lazily on the first
    // op, which happens after this function runs.
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        std::env::set_var("RAYON_NUM_THREADS", DEFAULT_CANDLE_THREADS.to_string());
    }
}

pub(crate) fn device_from_kind(kind: DeviceKind) -> Result<candle_core::Device, Error> {
    match kind {
        DeviceKind::Cpu => Ok(candle_core::Device::Cpu),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => Ok(candle_core::Device::new_cuda(0)?),
        #[cfg(not(feature = "cuda"))]
        DeviceKind::Cuda => Err(Error::UnsupportedBackend(
            "CUDA backend is not enabled; build with --features rocr/cuda".into(),
        )),
        #[cfg(feature = "metal")]
        DeviceKind::Metal => Ok(candle_core::Device::new_metal(0)?),
        #[cfg(not(feature = "metal"))]
        DeviceKind::Metal => Err(Error::UnsupportedBackend(
            "Metal backend is not enabled; build with --features rocr/metal".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_image(w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            w,
            h,
            image::Rgb([255, 255, 255]),
        ))
    }

    #[test]
    fn configure_candle_threads_sets_default() {
        // The env var is global; only assert the wiring when we can control it.
        std::env::remove_var("RAYON_NUM_THREADS");
        configure_candle_threads();
        let v = std::env::var("RAYON_NUM_THREADS").unwrap();
        assert_eq!(v, DEFAULT_CANDLE_THREADS.to_string());
    }

    #[test]
    fn crop_polygon_extracts_bounding_box() {
        let img = white_image(10, 10);
        let poly = [[2.0, 2.0], [6.0, 2.0], [6.0, 4.0], [2.0, 4.0]];
        let crop = crop_polygon(&img, &poly).unwrap();
        assert_eq!(crop.width(), 4, "crop width should span x∈[2,6)");
        assert_eq!(crop.height(), 2, "crop height should span y∈[2,4)");
    }

    #[test]
    fn crop_polygon_clamps_to_image_bounds() {
        let img = white_image(10, 10);
        // Polygon partially outside: the crop must clamp to the image edge.
        let poly = [[-3.0, -2.0], [5.0, -2.0], [5.0, 3.0], [-3.0, 3.0]];
        let crop = crop_polygon(&img, &poly).unwrap();
        assert_eq!(crop.width(), 5, "clamped to x∈[0,5)");
        assert_eq!(crop.height(), 3, "clamped to y∈[0,3)");
    }

    #[test]
    fn crop_polygon_fully_outside_is_none() {
        let img = white_image(10, 10);
        let poly = [[20.0, 20.0], [21.0, 20.0], [21.0, 21.0], [20.0, 21.0]];
        assert!(crop_polygon(&img, &poly).is_none());
    }

    #[test]
    fn crop_polygon_empty_is_none() {
        let img = white_image(10, 10);
        let poly: Vec<[f32; 2]> = vec![];
        assert!(crop_polygon(&img, &poly).is_none());
    }

    #[test]
    fn device_kind_reports_unsupported_backends() {
        #[cfg(not(feature = "cuda"))]
        assert!(matches!(
            device_from_kind(DeviceKind::Cuda),
            Err(Error::UnsupportedBackend(_))
        ));
        #[cfg(not(feature = "metal"))]
        assert!(matches!(
            device_from_kind(DeviceKind::Metal),
            Err(Error::UnsupportedBackend(_))
        ));
    }
}
