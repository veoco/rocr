//! LightSVTR recognition neck: local 1×7 depthwise conv + global self-attention.
//!
//! Structure (verified against the official ONNX graph):
//! ```text
//! pool = avgpool3x2(backbone_out)            # [B, C, 1, T']
//! a    = silu(BN(conv1x1(pool)))             # conv_block.0 (parallel branch)
//! b    = silu(BN(conv1x1(pool)))             # conv_block.1
//! c    = silu(BN(dw1x7(b)))                  # conv_block.2
//! x    = transpose(flatten(b + c))           # [B, T, H]
//! # 2 svtr blocks, each with additive skip:
//! x = x + attn(LN(x));  x = x + mlp(LN(x))
//! x = final LN(x)
//! # combine with the parallel conv branch before the head:
//! x = transpose(reshape(x)) + a
//! ```

use candle_core::Tensor;
use candle_nn::{BatchNorm, LayerNorm, Module, ModuleT};

use crate::error::Error;

use super::backbone::{conv2d_2d, Act};
use super::config::RecConfig;
use super::linear;
use super::weights::W;

const BN_EPS: f64 = 1e-5;
const ATTENTION_HEADS: usize = 8;

fn layer_norm(w: &W, prefix: &str, _hidden: usize, eps: f64) -> Result<LayerNorm, Error> {
    let weight = w.get(&format!("{prefix}.weight"))?;
    let bias = w.get(&format!("{prefix}.bias"))?;
    Ok(LayerNorm::new(weight, bias, eps))
}

/// A conv + BN + SiLU block (used by the neck).
struct ConvBnSilu {
    conv_w: Tensor,
    bn: BatchNorm,
}

impl ConvBnSilu {
    fn new(w: &W, prefix: &str, c_out: usize) -> Result<Self, Error> {
        let conv_w = w.get(&format!("{prefix}.convolution.weight"))?;
        let bn_w = w.get(&format!("{prefix}.normalization.weight"))?;
        let bn_b = w.get(&format!("{prefix}.normalization.bias"))?;
        let bn_m = w.get(&format!("{prefix}.normalization.running_mean"))?;
        let bn_v = w.get(&format!("{prefix}.normalization.running_var"))?;
        let bn = BatchNorm::new(c_out, bn_m, bn_v, bn_w, bn_b, BN_EPS)?;
        Ok(Self { conv_w, bn })
    }

    fn forward(&self, x: &Tensor, pad: (usize, usize), groups: usize) -> Result<Tensor, Error> {
        let y = conv2d_2d(x, &self.conv_w, None, pad, (1, 1), groups)?;
        let y = self.bn.forward_t(&y, false)?;
        Act::Silu.apply(&y)
    }
}

/// Multi-head self-attention (8 heads, `scale = 1/sqrt(head_dim)`).
struct Attention {
    qkv_w: Tensor,
    qkv_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    heads: usize,
}

impl Attention {
    fn new(w: &W, prefix: &str, heads: usize) -> Result<Self, Error> {
        Ok(Self {
            qkv_w: w.get(&format!("{prefix}.qkv.weight"))?,
            qkv_b: w.get(&format!("{prefix}.qkv.bias"))?,
            proj_w: w.get(&format!("{prefix}.projection.weight"))?,
            proj_b: w.get(&format!("{prefix}.projection.bias"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (b, t, hidden) = x.dims3()?;
        let head_dim = hidden / self.heads;
        let qkv = linear(x, &self.qkv_w, &self.qkv_b)?; // [B,T,3H]
        let qkv = qkv.reshape((b, t, 3, self.heads, head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // [3,B,heads,T,hd]
        let q = qkv.narrow(0, 0, 1)?.squeeze(0)?;
        let k = qkv.narrow(0, 1, 1)?.squeeze(0)?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(0)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let q = q.affine(scale, 0.0)?;
        let k = k.transpose(2, 3)?; // [B,heads,hd,T]
        let scores = q.matmul(&k)?; // [B,heads,T,T]
        let scores = candle_nn::ops::softmax(&scores, 3)?;
        let ctx = scores.matmul(&v)?; // [B,heads,T,hd]
        let ctx = ctx.permute((0, 2, 1, 3))?; // [B,T,heads,hd]
        let ctx = ctx.reshape((b, t, hidden))?;
        linear(&ctx, &self.proj_w, &self.proj_b)
    }
}

/// MLP: hidden → 2·hidden → hidden with SiLU.
struct Mlp {
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

impl Mlp {
    fn new(w: &W, prefix: &str) -> Result<Self, Error> {
        Ok(Self {
            fc1_w: w.get(&format!("{prefix}.fc1.weight"))?,
            fc1_b: w.get(&format!("{prefix}.fc1.bias"))?,
            fc2_w: w.get(&format!("{prefix}.fc2.weight"))?,
            fc2_b: w.get(&format!("{prefix}.fc2.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let h = linear(x, &self.fc1_w, &self.fc1_b)?;
        let h = h.silu()?;
        linear(&h, &self.fc2_w, &self.fc2_b)
    }
}

/// LightSVTR encoder.
pub struct LightSvtr {
    conv_block0: ConvBnSilu,
    conv_block1: ConvBnSilu,
    conv_block2: ConvBnSilu,
    hidden: usize,
    /// Layer norms in execution order:
    /// `[norm, b0.ln1, b0.ln2, b1.ln1, ...]` (the final `b{n-1}.ln2` is kept
    /// separately as `final_ln` with epsilon 1e-6).
    norms: Vec<LayerNorm>,
    attns: Vec<Attention>,
    mlps: Vec<Mlp>,
    final_ln: LayerNorm,
}

impl LightSvtr {
    pub fn new(w: &W, cfg: &RecConfig) -> Result<Self, Error> {
        let h = cfg.hidden_size;
        let n = cfg.depth;
        let conv_block0 = ConvBnSilu::new(w, "head.encoder.conv_block.0", h)?;
        let conv_block1 = ConvBnSilu::new(w, "head.encoder.conv_block.1", h)?;
        let conv_block2 = ConvBnSilu::new(w, "head.encoder.conv_block.2", h)?;
        let mut norms = Vec::with_capacity(2 * n);
        for i in 0..n {
            let base = format!("head.encoder.svtr_block.{i}");
            norms.push(layer_norm(w, &format!("{base}.layer_norm1"), h, 1e-5)?);
            norms.push(layer_norm(w, &format!("{base}.layer_norm2"), h, 1e-5)?);
        }
        let mut attns = Vec::with_capacity(n);
        let mut mlps = Vec::with_capacity(n);
        for i in 0..n {
            let base = format!("head.encoder.svtr_block.{i}");
            attns.push(Attention::new(
                w,
                &format!("{base}.self_attn"),
                ATTENTION_HEADS,
            )?);
            mlps.push(Mlp::new(w, &format!("{base}.mlp"))?);
        }
        // `head.encoder.norm` is the final pre-head norm (epsilon 1e-6).
        let final_ln = layer_norm(w, "head.encoder.norm", h, 1e-6)?;
        Ok(Self {
            conv_block0,
            conv_block1,
            conv_block2,
            hidden: h,
            norms,
            attns,
            mlps,
            final_ln,
        })
    }

    /// Debug: run up to the point before the svtr blocks; returns
    /// `(a, b_plus_c_bt)` where `a` is the parallel conv-branch output and
    /// `b_plus_c_bt` is the flattened `[B, T, H]` main path.
    #[allow(dead_code)]
    pub(crate) fn debug_pre_svtr(&self, x: &Tensor) -> Result<(Tensor, Tensor), Error> {
        let x = x.avg_pool2d_with_stride((3, 2), (3, 2))?;
        let a = self.conv_block0.forward(&x, (0, 0), 1)?;
        let b = self.conv_block1.forward(&x, (0, 0), 1)?;
        let c = self.conv_block2.forward(&b, (0, 3), self.hidden)?;
        let main = (b + c)?.squeeze(2)?.permute((0, 2, 1))?;
        Ok((a, main))
    }

    /// Debug: run the svtr blocks, returning features after each step:
    /// `[main, ln0, after_attn0, ln1, after_mlp0, ln2, after_attn1, ln3, after_mlp1]`.
    #[allow(dead_code)]
    pub(crate) fn debug_svtr(&self, x: &Tensor) -> Result<Vec<Tensor>, Error> {
        let x = x.avg_pool2d_with_stride((3, 2), (3, 2))?;
        let b = self.conv_block1.forward(&x, (0, 0), 1)?;
        let c = self.conv_block2.forward(&b, (0, 3), self.hidden)?;
        let mut x = (b + c)?.squeeze(2)?.permute((0, 2, 1))?;
        let mut steps = vec![x.clone()];
        for i in 0..self.attns.len() {
            let ln_out = self.norms[i * 2].forward(&x)?;
            steps.push(ln_out.clone());
            let attn = self.attns[i].forward(&ln_out)?;
            x = (x + attn)?;
            steps.push(x.clone());
            let ln_out = self.norms[i * 2 + 1].forward(&x)?;
            steps.push(ln_out.clone());
            let mlp = self.mlps[i].forward(&ln_out)?;
            x = (x + mlp)?;
            steps.push(x.clone());
        }
        Ok(steps)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // Backbone output is [B, C_last, 3, T]; the neck AveragePool reduces
        // H 3→1 and W→W/2 (kernel/stride (3,2)).
        let x = x.avg_pool2d_with_stride((3, 2), (3, 2))?; // [B, C, 1, T']
        let a = self.conv_block0.forward(&x, (0, 0), 1)?; // parallel branch, kept for head
        let b = self.conv_block1.forward(&x, (0, 0), 1)?;
        let c = self.conv_block2.forward(&b, (0, 3), self.hidden)?; // 1×7 depthwise
        let x = (b + c)?.squeeze(2)?.permute((0, 2, 1))?; // [B, T, H]
                                                          // svtr blocks with additive skip connections
        let mut x = x;
        for i in 0..self.attns.len() {
            let ln1 = &self.norms[i * 2];
            let ln2 = &self.norms[i * 2 + 1];
            let attn = self.attns[i].forward(&ln1.forward(&x)?)?;
            x = (x + attn)?;
            let mlp = self.mlps[i].forward(&ln2.forward(&x)?)?;
            x = (x + mlp)?;
        }
        // Final LN (eps=1e-6), then combine with the parallel conv branch.
        let x = self.final_ln.forward(&x)?;
        let (b, t, h) = x.dims3()?;
        let x = x.reshape((b, 1, t, h))?.permute((0, 3, 1, 2))?; // [B,H,1,T]
        let x = (x + a)?.squeeze(2)?.permute((0, 2, 1))?; // [B,T,H]
        Ok(x)
    }
}
