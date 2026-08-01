//! Document unwarping / rectification (UVDoc).
//!
//! UVDoc is a deep encoder–decoder that predicts a 2-channel sampling grid
//! from a `[1,3,712,488]` input and rectifies curved document pages. The
//! network is: a ResNet encoder (5×5 convs + dilated residual blocks) → a
//! bridge of dilated 3×3 convs (6 stages, whose outputs are concatenated) → a
//! head (1×1 fusion + 5×5 PReLU + 5×5 → 2 channels). The predicted grid is
//! resized to the original image size and applied via bilinear `grid_sample`
//! (align_corners, zeros padding) to produce the rectified image.
//!
//! Numerically verified against the official `PaddlePaddle/UVDoc_onnx` export
//! (see `scripts/verify_uvdoc_onnx.py` and the `unwarp_matches_onnx` test).

use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::{BatchNorm, ModuleT};
use image::{DynamicImage, RgbImage};

use crate::error::Error;
use crate::model_loader;
use crate::rec::weights::W;

const BN_EPS: f64 = 1e-5;

/// UVDoc fixed input size (H × W), matching `UVDocImageProcessor`.
pub const UNWARP_HEIGHT: usize = 712;
pub const UNWARP_WIDTH: usize = 488;

/// Activation applied after conv + BN (torch `ACT2FN`).
enum Act {
    None,
    Relu,
    /// PReLU with a scalar weight (torch `nn.PReLU(num_parameters=1)`).
    PreLu(Tensor),
}

fn act_apply(a: &Act, x: &Tensor) -> Result<Tensor, Error> {
    match a {
        Act::None => Ok(x.clone()),
        Act::Relu => Ok(x.relu()?),
        Act::PreLu(w) => {
            // PReLU: max(0,x) + w·min(0,x).
            let pos = x.relu()?;
            let neg = x.minimum(&Tensor::zeros_like(x)?)?;
            Ok((pos + neg.broadcast_mul(w)?)?)
        }
    }
}

/// Reflect-pad a `[N,C,H,W]` tensor by `pad` pixels on all four sides
/// (torch `ReflectionPad2d` semantics: out-of-bounds elements mirror the
/// interior without repeating the edge itself).
fn reflect_pad2d(x: &Tensor, pad: usize) -> Result<Tensor, Error> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let (_, _, _h, w) = x.dims4()?;
    let left = x.narrow(3, 1, pad)?.contiguous()?.flip(&[3])?;
    let right = x.narrow(3, w - 1 - pad, pad)?.contiguous()?.flip(&[3])?;
    let xw = Tensor::cat(&[left, x.clone(), right], 3)?;
    let (_, _, hw, _) = xw.dims4()?;
    let top = xw.narrow(2, 1, pad)?.contiguous()?.flip(&[2])?;
    let bottom = xw.narrow(2, hw - 1 - pad, pad)?.contiguous()?.flip(&[2])?;
    Ok(Tensor::cat(&[top, xw, bottom], 2)?)
}

/// Conv (optional bias) + BatchNorm + activation (torch `UVDocConvLayer`).
struct ConvLayer {
    w: Tensor,
    b: Option<Tensor>,
    bn: BatchNorm,
    act: Act,
}

impl ConvLayer {
    fn load(w: &W, prefix: &str, act: Act) -> Result<Self, Error> {
        let conv_w = w.get(&format!("{prefix}.convolution.weight"))?;
        let c_out = conv_w.dims()[0];
        let b = if w.has(&format!("{prefix}.convolution.bias")) {
            Some(w.get(&format!("{prefix}.convolution.bias"))?)
        } else {
            None
        };
        let bn = BatchNorm::new(
            c_out,
            w.get(&format!("{prefix}.normalization.running_mean"))?,
            w.get(&format!("{prefix}.normalization.running_var"))?,
            w.get(&format!("{prefix}.normalization.weight"))?,
            w.get(&format!("{prefix}.normalization.bias"))?,
            BN_EPS,
        )?;
        Ok(Self {
            w: conv_w,
            b,
            bn,
            act,
        })
    }

    /// Conv a `[N,C,H,W]` tensor with the given padding/stride/dilation.
    /// `reflect` reflection-pads the input first (zero padding is applied by
    /// the conv itself otherwise).
    fn apply(
        &self,
        x: &Tensor,
        pad: usize,
        stride: usize,
        dilation: usize,
        reflect: bool,
    ) -> Result<Tensor, Error> {
        let y = if reflect {
            reflect_pad2d(x, pad)?.conv2d(&self.w, 0, stride, dilation, 1)?
        } else {
            x.conv2d(&self.w, pad, stride, dilation, 1)?
        };
        let y = match &self.b {
            Some(b) => {
                let c = b.dims1()?;
                y.broadcast_add(&b.reshape((1, c, 1, 1))?)?
            }
            None => y,
        };
        let y = self.bn.forward_t(&y, false)?;
        act_apply(&self.act, &y)
    }
}

/// A dilated residual block: optional stride-2 `conv_down`, then `conv_start`
/// + `conv_final`, add + ReLU.
struct ResidualBlock {
    conv_down: Option<ConvLayer>,
    conv_start: ConvLayer,
    conv_final: ConvLayer,
    stride: usize,
    pad: usize,
    dilation: usize,
}

impl ResidualBlock {
    fn load(
        w: &W,
        prefix: &str,
        _in_c: usize,
        _out_c: usize,
        dilation: usize,
        downsample: bool,
    ) -> Result<Self, Error> {
        let conv_down = if downsample {
            Some(ConvLayer::load(
                w,
                &format!("{prefix}.conv_down"),
                Act::None,
            )?)
        } else {
            None
        };
        let conv_start = ConvLayer::load(w, &format!("{prefix}.conv_start"), Act::Relu)?;
        let conv_final = ConvLayer::load(w, &format!("{prefix}.conv_final"), Act::None)?;
        let stride = if downsample { 2 } else { 1 };
        let pad = dilation * 2;
        Ok(Self {
            conv_down,
            conv_start,
            conv_final,
            stride,
            pad,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let residual = match &self.conv_down {
            // 5×5, stride 2, padding = kernel//2 = 2.
            Some(cd) => cd.apply(x, 2, 2, 1, false)?,
            None => x.clone(),
        };
        let y = self
            .conv_start
            .apply(x, self.pad, self.stride, self.dilation, false)?;
        let y = self
            .conv_final
            .apply(&y, self.pad, 1, self.dilation, false)?;
        Ok((y + residual)?.relu()?)
    }
}

/// One bridge stage: a sequence of dilated 3×3 convs.
struct BridgeStage {
    blocks: Vec<ConvLayer>,
    pads: Vec<usize>,
    dilations: Vec<usize>,
}

/// The UVDoc network. Loaded from a `UVDoc_safetensors`-style repository.
pub struct UVDoc {
    resnet_head: Vec<ConvLayer>,
    stages: Vec<Vec<ResidualBlock>>,
    bridge: Vec<BridgeStage>,
    bridge_connector: ConvLayer,
    out_down: ConvLayer,
    out_up_w: Tensor,
    out_up_b: Tensor,
    device: Device,
}

impl UVDoc {
    /// Load the network from its repository directory.
    pub fn new(repo: &Path, device: &Device) -> Result<Self, Error> {
        let config = model_loader::load_json(repo, model_loader::CONFIG_FILE)?;
        let bc = &config["backbone_config"];

        let resnet_head_cfg: Vec<(usize, usize)> = bc["resnet_head"]
            .as_array()
            .ok_or_else(|| Error::Config("resnet_head missing".into()))?
            .iter()
            .map(|v| {
                (
                    v[0].as_u64().unwrap_or(3) as usize,
                    v[1].as_u64().unwrap_or(32) as usize,
                )
            })
            .collect();
        let resnet_configs = bc["resnet_configs"]
            .as_array()
            .ok_or_else(|| Error::Config("resnet_configs missing".into()))?;
        let stage_configs = bc["stage_configs"]
            .as_array()
            .ok_or_else(|| Error::Config("stage_configs missing".into()))?;

        let w = W(&model_loader::load_tensors(repo, device)?);

        // ResNet head: 5×5 stride-2 convs (padding = kernel//2), ReLU.
        let mut resnet_head = Vec::with_capacity(resnet_head_cfg.len());
        for i in 0..resnet_head_cfg.len() {
            let layer =
                ConvLayer::load(&w, &format!("backbone.resnet.resnet_head.{i}"), Act::Relu)?;
            resnet_head.push(layer);
        }

        // ResNet downsampling stages.
        let mut stages = Vec::with_capacity(resnet_configs.len());
        for (si, stage) in resnet_configs.iter().enumerate() {
            let blocks = stage
                .as_array()
                .ok_or_else(|| Error::Config("malformed resnet_configs stage".into()))?;
            let mut layers = Vec::with_capacity(blocks.len());
            for (li, blk) in blocks.iter().enumerate() {
                let in_c = blk[0].as_u64().unwrap_or(32) as usize;
                let out_c = blk[1].as_u64().unwrap_or(in_c as u64) as usize;
                let dilation = blk[2].as_u64().unwrap_or(1) as usize;
                let downsample = blk[3].as_bool().unwrap_or(false);
                let prefix = format!("backbone.resnet.resnet_down.{si}.layers.{li}");
                layers.push(ResidualBlock::load(
                    &w, &prefix, in_c, out_c, dilation, downsample,
                )?);
            }
            stages.push(layers);
        }

        // Bridge: 6 stages of dilated 3×3 convs (kernel 3, pad = dilation).
        let mut bridge = Vec::with_capacity(stage_configs.len());
        for (si, stage) in stage_configs.iter().enumerate() {
            let blocks = stage
                .as_array()
                .ok_or_else(|| Error::Config("malformed stage_configs stage".into()))?;
            let mut bs = BridgeStage {
                blocks: Vec::with_capacity(blocks.len()),
                pads: Vec::with_capacity(blocks.len()),
                dilations: Vec::with_capacity(blocks.len()),
            };
            for (li, blk) in blocks.iter().enumerate() {
                let dilation = blk[1].as_u64().unwrap_or(1) as usize;
                let prefix = format!("backbone.bridge.bridge.{si}.blocks.{li}");
                let conv = ConvLayer::load(&w, &prefix, Act::Relu)?;
                bs.blocks.push(conv);
                bs.pads.push(dilation);
                bs.dilations.push(dilation);
            }
            bridge.push(bs);
        }

        // Head: 1×1 bridge connector (768→128, ReLU), 5×5 PReLU conv (reflect),
        // and the final 5×5 conv to a 2-channel grid.
        let bridge_connector = ConvLayer::load(&w, "head.bridge_connector", Act::Relu)?;
        let prelu_w = w.get("head.out_point_positions2D.conv_down.activation.weight")?;
        let out_down = ConvLayer::load(
            &w,
            "head.out_point_positions2D.conv_down",
            Act::PreLu(prelu_w),
        )?;
        let out_up_w = w.get("head.out_point_positions2D.conv_up.weight")?;
        let out_up_b = w.get("head.out_point_positions2D.conv_up.bias")?;

        Ok(Self {
            resnet_head,
            stages,
            bridge,
            bridge_connector,
            out_down,
            out_up_w,
            out_up_b,
            device: device.clone(),
        })
    }

    /// Run the network on a `[1,3,712,488]` input, returning the sampling
    /// grid `[1,2,45,31]` (channel 0 = x, channel 1 = y, values in [-1, 1]).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let mut x = x.clone();
        for h in &self.resnet_head {
            x = h.apply(&x, 2, 2, 1, false)?;
        }
        for stage in &self.stages {
            for blk in stage {
                x = blk.forward(&x)?;
            }
        }
        // The bridge stages all process the ResNet output in parallel (each
        // stage output is concatenated along the channel dim).
        let mut feats = Vec::new();
        for stage in &self.bridge {
            let mut y = x.clone();
            for (i, blk) in stage.blocks.iter().enumerate() {
                y = blk.apply(&y, stage.pads[i], 1, stage.dilations[i], false)?;
            }
            feats.push(y);
        }
        let fused = Tensor::cat(&feats, 1)?;
        let y = self.bridge_connector.apply(&fused, 0, 1, 1, false)?;
        let y = self.out_down.apply(&y, 2, 1, 1, true)?;
        let y = reflect_pad2d(&y, 2)?.conv2d(&self.out_up_w, 0, 1, 1, 1)?;
        let c = self.out_up_b.dims1()?;
        Ok(y.broadcast_add(&self.out_up_b.reshape((1, c, 1, 1))?)?)
    }

    /// Rectify a document image: resize to the fixed network input, predict the
    /// sampling grid, then warp the original pixels through it. The output has
    /// the same size as the input.
    pub fn unwarp_image(&self, img: &DynamicImage) -> Result<DynamicImage, Error> {
        let (oh, ow) = (img.height() as usize, img.width() as usize);
        let x = to_uvdoc_input(img, &self.device)?;
        let grid = self.forward(&x)?;
        // Resize the coarse grid to the original size, then sample the original
        // pixels (align_corners bilinear grid_sample, matching the ONNX graph).
        let gv = grid.flatten_all()?.to_vec1::<f32>()?;
        let (_, _, gh, gw) = grid.dims4()?;
        let resized = resize_align_corners(&gv, 2, gh, gw, oh, ow);
        let grid = Tensor::from_vec(resized, (1, 2, oh, ow), grid.device())?;
        let grid = grid.permute((0, 2, 3, 1))?;
        let out = grid_sample2d(&image_to_tensor_raw(img, &self.device)?, &grid)?;
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        let mut rgb = vec![0u8; oh * ow * 3];
        for c in 0..3 {
            for y in 0..oh {
                for x in 0..ow {
                    rgb[(y * ow + x) * 3 + c] = v[(c * oh + y) * ow + x].clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(DynamicImage::ImageRgb8(
            RgbImage::from_raw(ow as u32, oh as u32, rgb)
                .ok_or_else(|| Error::Image("invalid unwarp output".into()))?,
        ))
    }
}

/// The source image as a `[1,3,H,W]` float tensor with pixel values in
/// `[0,255]` (no resize, no normalization).
fn image_to_tensor_raw(img: &DynamicImage, device: &Device) -> Result<Tensor, Error> {
    let rgb = img.to_rgb8();
    let (h, w) = (rgb.height() as usize, rgb.width() as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for (i, p) in rgb.as_raw().chunks_exact(3).enumerate() {
        chw[i] = p[0] as f32;
        chw[h * w + i] = p[1] as f32;
        chw[2 * h * w + i] = p[2] as f32;
    }
    Ok(Tensor::from_vec(chw, (1, 3, h, w), device)?)
}

/// Align-corners bilinear resize over a `[C,H,W]` row-major buffer, matching
/// the ONNX `Resize` (`coordinate_transformation_mode=align_corners`) exactly:
/// `src = dst * (in - 1) / (out - 1)` evaluated in float32 (multiply first),
/// with `floor` indexing and `min(x0+1, in-1)`.
fn resize_align_corners(
    src: &[f32],
    c: usize,
    sh: usize,
    sw: usize,
    dh: usize,
    dw: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; c * dh * dw];
    for y in 0..dh {
        let sy = (y as f32) * (sh - 1) as f32 / (dh - 1) as f32;
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = sy - y0 as f32;
        for x in 0..dw {
            let sx = (x as f32) * (sw - 1) as f32 / (dw - 1) as f32;
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = sx - x0 as f32;
            for cc in 0..c {
                let b = |yy: usize, xx: usize| src[(cc * sh + yy) * sw + xx];
                let top = (1.0 - wx) * b(y0, x0) + wx * b(y0, x1);
                let bottom = (1.0 - wx) * b(y1, x0) + wx * b(y1, x1);
                out[(cc * dh + y) * dw + x] = (1.0 - wy) * top + wy * bottom;
            }
        }
    }
    out
}

/// Resize the image to the fixed 712×488 UVDoc input with align-corners
/// bilinear interpolation (the ONNX graph's first `Resize`), returning a
/// `[1,3,712,488]` float tensor in `[0,255]`.
fn to_uvdoc_input(img: &DynamicImage, device: &Device) -> Result<Tensor, Error> {
    let rgb = img.to_rgb8();
    let (h, w) = (rgb.height() as usize, rgb.width() as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for (i, p) in rgb.as_raw().chunks_exact(3).enumerate() {
        chw[i] = p[0] as f32;
        chw[h * w + i] = p[1] as f32;
        chw[2 * h * w + i] = p[2] as f32;
    }
    let resized = resize_align_corners(&chw, 3, h, w, UNWARP_HEIGHT, UNWARP_WIDTH);
    Ok(Tensor::from_vec(
        resized,
        (1, 3, UNWARP_HEIGHT, UNWARP_WIDTH),
        device,
    )?)
}

/// Bilinear `grid_sample` with `align_corners=true` and `padding_mode=zeros`,
/// matching the ONNX `GridSample` node. `input` is `[1,C,H,W]`; `grid` is
/// `[1,H,W,2]` with `(x, y)` coordinates in `[-1,1]`. Returns `[1,C,H,W]`.
///
/// Implemented on the host for exact parity with torch/ONNX semantics.
#[doc(hidden)]
pub fn grid_sample2d(input: &Tensor, grid: &Tensor) -> Result<Tensor, Error> {
    let (_, c, h, w) = input.dims4()?;
    let (_, gh, gw, _) = grid.dims4()?;
    if gh != h || gw != w {
        return Err(Error::Config(format!(
            "grid {gh}x{gw} does not match input {h}x{w}"
        )));
    }
    let inp = input.flatten_all()?.to_vec1::<f32>()?;
    let grd = grid.flatten_all()?.to_vec1::<f32>()?;
    let (iw, ih) = (w as f32, h as f32);
    let mut out = vec![0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            let gx = grd[(y * w + x) * 2];
            let gy = grd[(y * w + x) * 2 + 1];
            // align_corners: map [-1,1] to [0,size-1].
            let px = (gx + 1.0) / 2.0 * (iw - 1.0);
            let py = (gy + 1.0) / 2.0 * (ih - 1.0);
            let x0 = px.floor();
            let y0 = py.floor();
            let x1 = x0 + 1.0;
            let y1 = y0 + 1.0;
            let wx = px - x0;
            let wy = py - y0;
            let in_x0 = x0 >= 0.0 && x0 <= iw - 1.0;
            let in_x1 = x1 >= 0.0 && x1 <= iw - 1.0;
            let in_y0 = y0 >= 0.0 && y0 <= ih - 1.0;
            let in_y1 = y1 >= 0.0 && y1 <= ih - 1.0;
            let (ix0, iy0) = (
                x0.clamp(0.0, iw - 1.0) as usize,
                y0.clamp(0.0, ih - 1.0) as usize,
            );
            let (ix1, iy1) = (
                x1.clamp(0.0, iw - 1.0) as usize,
                y1.clamp(0.0, ih - 1.0) as usize,
            );
            for cc in 0..c {
                let (b00, b01, b10, b11) = (
                    (cc * h + iy0) * w + ix0,
                    (cc * h + iy0) * w + ix1,
                    (cc * h + iy1) * w + ix0,
                    (cc * h + iy1) * w + ix1,
                );
                let v00 = if in_x0 && in_y0 { inp[b00] } else { 0.0 };
                let v01 = if in_x1 && in_y0 { inp[b01] } else { 0.0 };
                let v10 = if in_x0 && in_y1 { inp[b10] } else { 0.0 };
                let v11 = if in_x1 && in_y1 { inp[b11] } else { 0.0 };
                let top = (1.0 - wx) * v00 + wx * v01;
                let bottom = (1.0 - wx) * v10 + wx * v11;
                out[(cc * h + y) * w + x] = (1.0 - wy) * top + wy * bottom;
            }
        }
    }
    Ok(Tensor::from_vec(out, (1, c, h, w), input.device())?)
}
