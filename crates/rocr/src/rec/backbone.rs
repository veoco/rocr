//! LCNetV4 (MetaFormer-style) backbone in recognition mode.
//!
//! Weight layout is the PP-OCRv6 safetensors convention:
//! `model.backbone.encoder.blocks.{stage}.blocks.{block}.{...}`.
//!
//! Block equation (verified against the official ONNX graph):
//! - stride-1 block:  `y = ch2(gelu(ch1(SE(DW(x))))) + SE(DW(x))` (SE optional)
//! - downsampling:    `y = ch2(gelu(ch1(BN(DW(x)))))`  (no residual, shape change)

use candle_core::Tensor;
use candle_nn::{BatchNorm, ModuleT};

use crate::error::Error;

use super::config::{BackboneConfig, BlockConfig};
use super::weights::W;

const BN_EPS: f64 = 1e-5;

/// Pad a `[B, C, H, W]` tensor by `(top, bottom)` on H and `(left, right)` on W.
pub(crate) fn pad4(
    x: &Tensor,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
) -> Result<Tensor, Error> {
    let x = x.pad_with_zeros(2, top, bottom)?;
    Ok(x.pad_with_zeros(3, left, right)?)
}

/// Convolution with per-dimension padding and (optionally asymmetric) stride.
///
/// candle's `conv2d` only accepts a scalar stride/padding, so:
/// - asymmetric padding (`pad_h != pad_w`) is applied manually first;
/// - an asymmetric stride `(2, 1)` is implemented as a stride-1 conv followed
///   by subsampling every other row (`out[i] = conv_s1[2i]`, mathematically
///   identical).
pub(crate) fn conv2d_2d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    pad: (usize, usize),
    stride: (usize, usize),
    groups: usize,
) -> Result<Tensor, Error> {
    let (pad_h, pad_w) = pad;
    let (sh, sw) = stride;
    let inp = if pad_h == pad_w {
        x.clone()
    } else {
        pad4(x, pad_h, pad_h, pad_w, pad_w)?
    };
    let pad = pad_h.min(pad_w);
    let asymmetric = sh != sw;
    let mut y = if asymmetric {
        inp.conv2d(w, pad, 1, 1, groups)?
    } else {
        inp.conv2d(w, pad, sh, 1, groups)?
    };
    if let Some(b) = bias {
        let c = b.dims1()?;
        y = y.broadcast_add(&b.reshape((1, c, 1, 1))?)?;
    }
    if asymmetric && sh == 2 {
        let (n, c, h, w) = y.dims4()?;
        // Pad to an even number of rows when needed so the reshape below is
        // valid; the padded row is never sampled.
        let h2 = h.div_ceil(2);
        let padded = if h % 2 == 1 {
            pad4(&y, 0, 1, 0, 0)?
        } else {
            y.clone()
        };
        y = padded
            .reshape((n, c, h2, 2, w))?
            .narrow(3, 0, 1)?
            .squeeze(3)?;
    }
    if asymmetric && sw == 2 {
        let (n, c, h, w) = y.dims4()?;
        let w2 = w.div_ceil(2);
        let padded = if w % 2 == 1 {
            pad4(&y, 0, 0, 0, 1)?
        } else {
            y.clone()
        };
        y = padded
            .reshape((n, c, h, w2, 2))?
            .narrow(4, 0, 1)?
            .squeeze(4)?;
    }
    Ok(y)
}

/// Activation applied after a conv+BN.
#[derive(Clone, Copy)]
pub(crate) enum Act {
    None,
    Relu,
    Gelu,
    Silu,
}

impl Act {
    pub(crate) fn apply(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(match self {
            Act::None => x.clone(),
            Act::Relu => x.relu()?,
            Act::Gelu => x.gelu_erf()?,
            Act::Silu => x.silu()?,
        })
    }
}

/// A 1×1 (or arbitrary) conv followed by batch norm and an activation.
struct ConvBn {
    conv_w: Tensor,
    bn: BatchNorm,
}

impl ConvBn {
    fn new(w: &W, prefix: &str, c_out: usize) -> Result<Self, Error> {
        let conv_w = w.get(&format!("{prefix}.convolution.weight"))?;
        let bn_w = w.get(&format!("{prefix}.normalization.weight"))?;
        let bn_b = w.get(&format!("{prefix}.normalization.bias"))?;
        let bn_m = w.get(&format!("{prefix}.normalization.running_mean"))?;
        let bn_v = w.get(&format!("{prefix}.normalization.running_var"))?;
        let bn = BatchNorm::new(c_out, bn_m, bn_v, bn_w, bn_b, BN_EPS)?;
        Ok(Self { conv_w, bn })
    }

    fn forward(
        &self,
        x: &Tensor,
        pad: (usize, usize),
        stride: (usize, usize),
        groups: usize,
        act: Act,
    ) -> Result<Tensor, Error> {
        let y = conv2d_2d(x, &self.conv_w, None, pad, stride, groups)?;
        let y = self.bn.forward_t(&y, false)?;
        act.apply(&y)
    }
}

/// Squeeze-and-excitation channel attention.
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

enum TokenConv {
    /// Fused (reparameterized) depthwise conv with bias.
    Fused { w: Tensor, b: Tensor },
    /// Non-fused depthwise conv + batch norm (used when stride != 1).
    ConvBn(ConvBn),
}

struct Block {
    token: TokenConv,
    ch1: ConvBn,
    ch2: ConvBn,
    se: Option<Se>,
    stride: (usize, usize),
    groups: usize,
    has_residual: bool,
}

impl Block {
    fn new(w: &W, cfg: &BlockConfig, prefix: &str) -> Result<Self, Error> {
        // The token depthwise conv is either a fused conv (weight+bias) or a
        // conv+BN (convolution.weight + normalization.*). Which form is used
        // varies by tier/block, so dispatch on the presence of BN weights
        // rather than on the stride.
        let token = if w.has(&format!("{prefix}.token_conv.normalization.weight")) {
            TokenConv::ConvBn(ConvBn::new(w, &format!("{prefix}.token_conv"), cfg.in_ch)?)
        } else {
            let tw = w.get(&format!("{prefix}.token_conv.weight"))?;
            let tb = w.get(&format!("{prefix}.token_conv.bias"))?;
            TokenConv::Fused { w: tw, b: tb }
        };
        let se = if cfg.use_se {
            Some(Se::new(w, &format!("{prefix}.token_squeeze_excitation"))?)
        } else {
            None
        };
        let ch1 = ConvBn::new(w, &format!("{prefix}.channel_conv1"), cfg.in_ch * 2)?;
        let ch2 = ConvBn::new(w, &format!("{prefix}.channel_conv2"), cfg.out_ch)?;
        // Residual only when the channel count is preserved and stride is 1
        // (matches the torch `has_residual = in==out and stride==1`).
        let has_residual = cfg.stride == (1, 1) && cfg.in_ch == cfg.out_ch;
        Ok(Self {
            token,
            ch1,
            ch2,
            se,
            stride: cfg.stride,
            groups: cfg.in_ch,
            has_residual,
        })
    }

    fn token_forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // kernel is always 3×3 for PP-OCRv6 blocks → pad (1,1).
        match &self.token {
            TokenConv::Fused { w, b } => conv2d_2d(x, w, Some(b), (1, 1), self.stride, self.groups),
            TokenConv::ConvBn(cb) => {
                let y = conv2d_2d(x, &cb.conv_w, None, (1, 1), self.stride, self.groups)?;
                Ok(cb.bn.forward_t(&y, false)?)
            }
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let t = self.token_forward(x)?;
        let t = match &self.se {
            Some(se) => se.forward(&t)?,
            None => t,
        };
        let h = self.ch1.forward(&t, (0, 0), (1, 1), 1, Act::Gelu)?;
        let h = self.ch2.forward(&h, (0, 0), (1, 1), 1, Act::None)?;
        if self.has_residual {
            Ok((h + t)?)
        } else {
            Ok(h)
        }
    }
}

enum Stem {
    Large {
        stem1: ConvBn,
        stem2a: ConvBn,
        stem2b: ConvBn,
        stem3: ConvBn,
        stem4: ConvBn,
    },
    Small {
        conv1: ConvBn,
        conv2: ConvBn,
    },
}

impl Stem {
    fn new(w: &W, cfg: &BackboneConfig) -> Result<Self, Error> {
        let prefix = "model.backbone.encoder.convolution";
        let (_c0, c1, c2) = cfg.stem_channels;
        if cfg.stem_type == "small" {
            Ok(Stem::Small {
                conv1: ConvBn::new(w, &format!("{prefix}.conv1"), c1)?,
                conv2: ConvBn::new(w, &format!("{prefix}.conv2"), c2)?,
            })
        } else {
            let stem2_ch = c1 / 2;
            Ok(Stem::Large {
                stem1: ConvBn::new(w, &format!("{prefix}.stem1"), c1)?,
                stem2a: ConvBn::new(w, &format!("{prefix}.stem2a"), stem2_ch)?,
                stem2b: ConvBn::new(w, &format!("{prefix}.stem2b"), c1)?,
                stem3: ConvBn::new(w, &format!("{prefix}.stem3"), c1)?,
                stem4: ConvBn::new(w, &format!("{prefix}.stem4"), c2)?,
            })
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        match self {
            Stem::Large {
                stem1,
                stem2a,
                stem2b,
                stem3,
                stem4,
            } => {
                // stem1: 3×3 stride-2
                let x = stem1.forward(x, (1, 1), (2, 2), 1, Act::Relu)?;
                // parallel branch: max pool (SAME_UPPER, k2 s1 → pad bottom/right)
                // + two 2×2 SAME_UPPER convs.
                let pool = pad4(&x, 0, 1, 0, 1)?.max_pool2d_with_stride((2, 2), (1, 1))?;
                let b = stem2_same(stem2a, &x)?;
                let b = stem2_same(stem2b, &b)?;
                let x = Tensor::cat(&[pool, b], 1)?;
                // stem3: 3×3 stride-2, then stem4: 1×1
                let x = stem3.forward(&x, (1, 1), (2, 2), 1, Act::Relu)?;
                stem4.forward(&x, (0, 0), (1, 1), 1, Act::Relu)
            }
            Stem::Small { conv1, conv2 } => {
                // conv1 (3×3 s2) + GELU, conv2 (3×3 s2), no final activation.
                let x = conv1.forward(x, (1, 1), (2, 2), 1, Act::Gelu)?;
                conv2.forward(&x, (1, 1), (2, 2), 1, Act::None)
            }
        }
    }
}

/// 2×2 conv with SAME_UPPER padding (pad bottom/right by 1, then conv 0).
fn stem2_same(cb: &ConvBn, x: &Tensor) -> Result<Tensor, Error> {
    let p = pad4(x, 0, 1, 0, 1)?;
    cb.forward(&p, (0, 0), (1, 1), 1, Act::Relu)
}

/// LCNetV4 backbone (recognition mode).
pub struct Backbone {
    stem: Stem,
    stages: Vec<Vec<Block>>,
}

impl Backbone {
    pub fn new(w: &W, cfg: &BackboneConfig) -> Result<Self, Error> {
        let stem = Stem::new(w, cfg)?;
        let mut stages = Vec::new();
        for (si, stage) in cfg.stages.iter().enumerate() {
            let mut blocks = Vec::new();
            for (bi, bc) in stage.iter().enumerate() {
                let prefix = format!("model.backbone.encoder.blocks.{si}.blocks.{bi}");
                blocks.push(Block::new(w, bc, &prefix)?);
            }
            stages.push(blocks);
        }
        Ok(Self { stem, stages })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let outs = self.forward_stages(x)?;
        outs.last()
            .cloned()
            .ok_or_else(|| Error::Config("no backbone stages".into()))
    }

    /// Run the backbone and return each stage's output feature map.
    /// Stage i is at downsampling factor 2^(i+2) (stem is 2×).
    pub fn forward_stages(&self, x: &Tensor) -> Result<Vec<Tensor>, Error> {
        let mut x = self.stem.forward(x)?;
        let mut outs = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x)?;
            }
            outs.push(x.clone());
        }
        Ok(outs)
    }

    /// Run the stem only (exposed for structural oracle tests).
    #[allow(dead_code)]
    pub(crate) fn stem_forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        self.stem.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    fn rand(n: usize, c: usize, h: usize, w: usize) -> Tensor {
        // Deterministic pseudo-random data in [0,1).
        let v: Vec<f32> = (0..n * c * h * w)
            .map(|i| (i as f32 * 7919.0).fract())
            .collect();
        Tensor::from_vec(v, (n, c, h, w), &Device::Cpu).unwrap()
    }

    fn weight(oc: usize, ic: usize, k: usize) -> Tensor {
        let v: Vec<f32> = (0..oc * ic * k * k)
            .map(|i| (i as f32 * 104729.0).fract())
            .collect();
        Tensor::from_vec(v, (oc, ic, k, k), &Device::Cpu).unwrap()
    }

    #[test]
    fn asymmetric_stride_h2_output_shape() {
        let x = rand(1, 3, 8, 8);
        let w = weight(4, 3, 3);
        let y = conv2d_2d(&x, &w, None, (0, 0), (2, 1), 1).unwrap();
        assert_eq!(
            y.dims(),
            &[1, 4, 3, 6],
            "8x8 stride1->6x6, even rows -> 3x6"
        );
    }

    #[test]
    fn asymmetric_stride_w2_output_shape() {
        let x = rand(1, 3, 8, 8);
        let w = weight(4, 3, 3);
        let y = conv2d_2d(&x, &w, None, (0, 0), (1, 2), 1).unwrap();
        assert_eq!(
            y.dims(),
            &[1, 4, 6, 3],
            "8x8 stride1->6x6, even cols -> 6x3"
        );
    }

    #[test]
    fn odd_dimensions_do_not_panic() {
        // Regression: previously reshape(h/2, 2) panicked on odd H/W.
        let x = rand(1, 3, 5, 6);
        let w = weight(4, 3, 3);
        let y = conv2d_2d(&x, &w, None, (0, 0), (2, 1), 1).unwrap();
        assert_eq!(
            y.dims(),
            &[1, 4, 2, 4],
            "5x6 stride1->3x4, even rows -> 2x4"
        );
        let y2 = conv2d_2d(&x, &w, None, (0, 0), (1, 2), 1).unwrap();
        assert_eq!(
            y2.dims(),
            &[1, 4, 3, 2],
            "5x6 stride1->3x4, even cols -> 3x2"
        );
    }

    #[test]
    fn asymmetric_stride_equals_subsampled_stride1() {
        let x = rand(1, 3, 8, 8);
        let w = weight(4, 3, 3);
        let y = conv2d_2d(&x, &w, None, (1, 1), (2, 1), 1).unwrap();
        // Reference: stride-1 conv then take even rows.
        let s1 = conv2d_2d(&x, &w, None, (1, 1), (1, 1), 1).unwrap();
        let h2 = s1.dims()[2].div_ceil(2);
        let ref_y = s1
            .reshape((1, 4, h2, 2, 8))
            .unwrap()
            .narrow(3, 0, 1)
            .unwrap()
            .squeeze(3)
            .unwrap();
        let d = (y - &ref_y).unwrap().abs().unwrap().max_all().unwrap();
        assert!(d.to_scalar::<f32>().unwrap() < 1e-5, "diff {d:?}");
    }

    #[test]
    fn asymmetric_padding_keeps_shape() {
        let x = rand(1, 3, 6, 6);
        let w = weight(4, 3, 3);
        // pad_h=2, pad_w=0: (2,2) vs (0,0). Output height grows, width shrinks.
        let y = conv2d_2d(&x, &w, None, (2, 0), (1, 1), 1).unwrap();
        assert_eq!(y.dims(), &[1, 4, 8, 4]);
    }
}
