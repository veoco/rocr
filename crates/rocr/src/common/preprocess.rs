//! Shared image → tensor preprocessing.
//!
//! The PP-OCRv6 transformers pipeline uses different normalization per task:
//! - recognition: RGB, scale 1/255, mean/std `[0.5,0.5,0.5]`
//! - detection:   BGR, scale 1/255, mean/std `[0.406,0.456,0.485]/[0.225,0.224,0.229]`
//!
//! Both were confirmed empirically against the official ONNX exports (see
//! `scripts/verify_onnx.py`).

use candle_core::{Device, Tensor};
use image::{DynamicImage, GenericImageView};

use crate::error::Error;

/// Normalization + layout parameters for an image → tensor transform.
#[derive(Debug, Clone, Copy)]
pub struct Normalize {
    /// Interpret the source image as BGR (swap R/B) instead of RGB.
    pub bgr: bool,
    /// Pixel scale factor (usually `1/255`).
    pub scale: f32,
    /// Per-channel mean, indexed by *output* channel order.
    pub mean: [f32; 3],
    /// Per-channel std, indexed by *output* channel order.
    pub std: [f32; 3],
}

/// Recognition normalization (RGB, `[0.5,0.5,0.5]`).
pub const REC_NORMALIZE: Normalize = Normalize {
    bgr: false,
    scale: 1.0 / 255.0,
    mean: [0.5, 0.5, 0.5],
    std: [0.5, 0.5, 0.5],
};

/// Detection normalization (BGR, ImageNet in BGR order).
pub const DET_NORMALIZE: Normalize = Normalize {
    bgr: true,
    scale: 1.0 / 255.0,
    mean: [0.406, 0.456, 0.485],
    std: [0.225, 0.224, 0.229],
};

/// Bilinear resize from a packed `RGB8` source to a `f32 [0,255]` HWC buffer,
/// using the same coordinate convention as OpenCV's `INTER_LINEAR`
/// (`src = (dst + 0.5) * scale - 0.5`, clamped).
fn bilinear_resize(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * dh * 3];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for i in 0..dh {
        let src_y = ((i as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = src_y.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = src_y - y0 as f32;
        for j in 0..dw {
            let src_x = ((j as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = src_x - x0 as f32;
            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let top = p00 + wx * (p01 - p00);
                let bot = p10 + wx * (p11 - p10);
                out[(i * dw + j) * 3 + c] = top + wy * (bot - top);
            }
        }
    }
    out
}

/// Resize an image to exactly `w × h` (bilinear) and produce a normalized
/// NCHW f32 tensor of shape `(1, 3, h, w)`.
pub fn image_to_tensor(
    img: &DynamicImage,
    w: usize,
    h: usize,
    norm: &Normalize,
    device: &Device,
) -> Result<Tensor, Error> {
    let rgb = img.to_rgb8();
    let (sw, sh) = (rgb.width() as usize, rgb.height() as usize);
    let resized = bilinear_resize(rgb.as_raw(), sw, sh, w, h);
    // Output channel c reads source channel `chan_of[c]` (handle BGR swap).
    let chan_of = if norm.bgr { [2, 1, 0] } else { [0, 1, 2] };
    let n = h * w;
    let mut flat = Vec::with_capacity(3 * n);
    for (c, &out_c) in chan_of.iter().enumerate() {
        for px in resized.chunks_exact(3) {
            let v = px[out_c] * norm.scale;
            flat.push((v - norm.mean[c]) / norm.std[c]);
        }
    }
    let t = Tensor::from_vec(flat, (1, 3, h, w), &Device::Cpu)?.to_device(device)?;
    Ok(t)
}

/// Recognition target height and maximum width.
pub const REC_HEIGHT: usize = 48;
pub const REC_MAX_WIDTH: usize = 3200;

/// Resize a text-line image for recognition: height exactly 48, width
/// proportional (rounded), capped at [`REC_MAX_WIDTH`]. Returns the tensor
/// and the target width.
pub fn rec_preprocess(img: &DynamicImage, device: &Device) -> Result<(Tensor, usize), Error> {
    let (w0, h0) = img.dimensions();
    let tw = ((REC_HEIGHT as u64 * w0 as u64 + u64::from(h0) / 2) / u64::from(h0)) as usize;
    let tw = tw.clamp(1, REC_MAX_WIDTH);
    let t = image_to_tensor(img, tw, REC_HEIGHT, &REC_NORMALIZE, device)?;
    Ok((t, tw))
}

/// Detection resize limits (from the official det preprocessor config).
pub const DET_LIMIT_SIDE_LEN: usize = 736;
pub const DET_MAX_SIDE: usize = 4000;
pub const DET_MULTIPLE: usize = 32;

/// Compute the resized `(h, w)` for detection input (DetResizeForTest:
/// `limit_side_len=736, limit_type=min`, rounded to a multiple of 32).
pub fn det_resize_dims(h0: usize, w0: usize) -> (usize, usize) {
    let min_side = h0.min(w0);
    let max_side = h0.max(w0) as f32;
    let mut ratio = if min_side < DET_LIMIT_SIDE_LEN {
        DET_LIMIT_SIDE_LEN as f32 / min_side as f32
    } else {
        1.0
    };
    if max_side * ratio > DET_MAX_SIDE as f32 {
        ratio = DET_MAX_SIDE as f32 / max_side;
    }
    let nh = ((h0 as f32 * ratio / DET_MULTIPLE as f32).round() as usize * DET_MULTIPLE)
        .max(DET_MULTIPLE);
    let nw = ((w0 as f32 * ratio / DET_MULTIPLE as f32).round() as usize * DET_MULTIPLE)
        .max(DET_MULTIPLE);
    (nh, nw)
}

/// Preprocess a full image for detection. Returns the tensor `(1,3,h,w)` and
/// the resized `(h, w)` (needed to map boxes back to the original image).
pub fn det_preprocess(
    img: &DynamicImage,
    device: &Device,
) -> Result<(Tensor, usize, usize), Error> {
    let (w0, h0) = img.dimensions();
    let (nh, nw) = det_resize_dims(h0 as usize, w0 as usize);
    let t = image_to_tensor(img, nw, nh, &DET_NORMALIZE, device)?;
    Ok((t, nh, nw))
}

/// Detection preprocessing with custom normalization (experiments).
pub fn det_preprocess_norm(
    img: &DynamicImage,
    device: &Device,
    bgr: bool,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<(Tensor, usize, usize), Error> {
    let norm = Normalize {
        bgr,
        scale: 1.0 / 255.0,
        mean: *mean,
        std: *std,
    };
    let (w0, h0) = img.dimensions();
    let (nh, nw) = det_resize_dims(h0 as usize, w0 as usize);
    let t = image_to_tensor(img, nw, nh, &norm, device)?;
    Ok((t, nh, nw))
}

/// Document-orientation classification resize limits (PP-LCNet_x1_0_doc_ori).
pub const DOC_RESIZE_SHORT: usize = 256;
pub const DOC_CROP: usize = 224;

/// Preprocess a full page image for document-orientation classification:
/// scale so the shorter edge is [`DOC_RESIZE_SHORT`], center-crop to
/// [`DOC_CROP`]×[`DOC_CROP`], then BGR normalize (mirrors the transformers
/// `PPLCNetImageProcessor` with `resize_short=256`, `do_center_crop=true`).
pub fn doc_preprocess(img: &DynamicImage, device: &Device) -> Result<Tensor, Error> {
    let (w0, h0) = img.dimensions();
    let scale = DOC_RESIZE_SHORT as f32 / h0.min(w0) as f32;
    let nh = (h0 as f32 * scale).round() as usize;
    let nw = (w0 as f32 * scale).round() as usize;
    let rgb = img.to_rgb8();
    let resized = bilinear_resize(rgb.as_raw(), w0 as usize, h0 as usize, nw, nh);
    let top = (nh - DOC_CROP) / 2;
    let left = (nw - DOC_CROP) / 2;
    // Output channel c reads source channel `chan_of[c]` (BGR swap).
    let chan_of = [2, 1, 0];
    let n = DOC_CROP * DOC_CROP;
    let mut flat = Vec::with_capacity(3 * n);
    for (c, &out_c) in chan_of.iter().enumerate() {
        for i in 0..DOC_CROP {
            for j in 0..DOC_CROP {
                let src = resized[((top + i) * nw + left + j) * 3 + out_c];
                let v = src * (1.0 / 255.0);
                flat.push((v - DET_NORMALIZE.mean[c]) / DET_NORMALIZE.std[c]);
            }
        }
    }
    let t = Tensor::from_vec(flat, (1, 3, DOC_CROP, DOC_CROP), &Device::Cpu)?.to_device(device)?;
    Ok(t)
}

/// Common max-absolute-difference between two f32 tensors (for tests).
pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32, Error> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    Ok(a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank_img(w: u32, h: u32) -> DynamicImage {
        DynamicImage::new_rgb8(w, h)
    }

    #[test]
    fn det_resize_no_upscale_when_large() {
        // 2000x3000: min side 2000 > 736 and max 3000 ≤ 4000 → ratio stays 1.
        let (nh, nw) = det_resize_dims(2000, 3000);
        assert_eq!((nh, nw), (2016, 3008)); // 2000/32=62.5→63*32=2016; 3000/32=93.75→94*32=3008
    }

    #[test]
    fn det_resize_upscales_to_736() {
        let (nh, nw) = det_resize_dims(700, 900);
        // ratio = 736/700 = 1.0514; nh=736, nw=round(946.3/32)*32=960
        assert_eq!((nh, nw), (736, 960));
    }

    #[test]
    fn rec_width_rounding() {
        // 400x60 → round(48*400/60) = round(320) = 320
        let img = blank_img(400, 60);
        let (t, tw) = rec_preprocess(&img, &Device::Cpu).unwrap();
        assert_eq!(tw, 320);
        assert_eq!(t.dims(), &[1, 3, 48, 320]);
    }

    #[test]
    fn rec_width_rounding_long() {
        // 700x60 → round(48*700/60) = round(560) = 560
        let img = blank_img(700, 60);
        let (_, tw) = rec_preprocess(&img, &Device::Cpu).unwrap();
        assert_eq!(tw, 560);
    }

    #[test]
    fn normalize_values_rgb() {
        // white pixel under RGB [0.5,0.5,0.5] → (1 - 0.5)/0.5 = 1.0
        let mut img = DynamicImage::new_rgb8(2, 1);
        for (px, val) in img.as_mut_rgb8().unwrap().pixels_mut().zip([255u8, 255u8]) {
            *px = image::Rgb([val, val, val]);
        }
        let t = image_to_tensor(&img, 2, 1, &REC_NORMALIZE, &Device::Cpu).unwrap();
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|&x| (x - 1.0).abs() < 1e-5), "{v:?}");
    }

    #[test]
    fn bgr_swaps_channels() {
        // pure red pixel (255,0,0); BGR normalize mean=[0.406,0.456,0.485]:
        // output channel 0 (B) = (0 - 0.406)/0.225, channel 2 (R) = (1-0.485)/0.229
        let mut img = DynamicImage::new_rgb8(1, 1);
        img.as_mut_rgb8()
            .unwrap()
            .put_pixel(0, 0, image::Rgb([255, 0, 0]));
        let t = image_to_tensor(&img, 1, 1, &DET_NORMALIZE, &Device::Cpu).unwrap();
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected_c0 = (0.0f32 - 0.406) / 0.225;
        let expected_c1 = (0.0f32 - 0.456) / 0.224;
        let expected_c2 = (1.0f32 - 0.485) / 0.229;
        assert!((v[0] - expected_c0).abs() < 1e-4, "c0={}", v[0]);
        assert!((v[1] - expected_c1).abs() < 1e-4, "c1={}", v[1]);
        assert!((v[2] - expected_c2).abs() < 1e-4, "c2={}", v[2]);
    }
}
