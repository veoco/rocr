//! Medium-tier detection neck: PAN (path aggregation) + intra-class blocks.
//!
//! Structure (from the PP-OCRv6 medium det transformers model):
//! ```text
//! adj[i]   = 1×1 conv(stage[i])                      # stage_ch → neck_out
//! top_down[3] = adj[3]; top_down[i] = adj[i] + up2(top_down[i+1])
//! proj[i]  = 9×9 conv(top_down[i])                   # → neck_out/4
//! bottom_up[0] = proj[0]; bottom_up[i] = proj[i] + 3×3 s2(bottom_up[i-1])
//! lateral[i] = 9×9 conv(proj[0] if i==0 else bottom_up[i])
//! intra[i] = intraclass_block[i](lateral[i])
//! out = cat(up8(intra[3]), up4(intra[2]), up2(intra[1]), intra[0])
//! ```

use candle_core::Tensor;
use candle_nn::{BatchNorm, ModuleT};

use crate::error::Error;
use crate::rec::backbone::conv2d_2d;
use crate::rec::weights::W;

use super::repfpn::DbHead;

const BN_EPS: f64 = 1e-5;

/// Medium det neck configuration.
pub struct MediumDetConfig {
    pub scale_factor_list: Vec<usize>,
    pub intraclass_block_number: usize,
}

impl MediumDetConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, Error> {
        let arr = |k: &str| -> Result<Vec<usize>, Error> {
            v.get(k)
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|e| e.as_u64())
                        .map(|e| e as usize)
                        .collect()
                })
                .ok_or_else(|| Error::Config(format!("missing {k}")))
        };
        Ok(Self {
            scale_factor_list: arr("scale_factor_list")?,
            intraclass_block_number: v
                .get("intraclass_block_number")
                .and_then(|x| x.as_u64())
                .unwrap_or(4) as usize,
        })
    }
}

/// A plain 2D conv with optional bias. Padding = kernel_size // 2.
struct Conv {
    w: Tensor,
    b: Option<Tensor>,
}

impl Conv {
    fn new(w: &W, name: &str, has_bias: bool) -> Result<Self, Error> {
        let w_ = w.get(&format!("{name}.weight"))?;
        let b = if has_bias {
            Some(w.get(&format!("{name}.bias"))?)
        } else {
            None
        };
        Ok(Self { w: w_, b })
    }

    fn forward(&self, x: &Tensor, stride: (usize, usize)) -> Result<Tensor, Error> {
        let (_, _, kh, kw) = self.w.dims4()?;
        let pad = (kh / 2, kw / 2);
        conv2d_2d(x, &self.w, self.b.as_ref(), pad, stride, 1)
    }
}

/// Conv + BN (no activation).
struct ConvBn {
    w: Tensor,
    bn: BatchNorm,
}

impl ConvBn {
    fn new(w: &W, prefix: &str) -> Result<Self, Error> {
        let w_ = w.get(&format!("{prefix}.convolution.weight"))?;
        let out = w_.dims()[0];
        let bn = BatchNorm::new(
            out,
            w.get(&format!("{prefix}.norm.running_mean"))?,
            w.get(&format!("{prefix}.norm.running_var"))?,
            w.get(&format!("{prefix}.norm.weight"))?,
            w.get(&format!("{prefix}.norm.bias"))?,
            BN_EPS,
        )?;
        Ok(Self { w: w_, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = conv2d_2d(x, &self.w, None, (0, 0), (1, 1), 1)?;
        Ok(self.bn.forward_t(&y, false)?)
    }
}

/// The intra-class block: reduce → 3 stages of summed asymmetric convs →
/// final conv, with a residual connection.
struct IntraclassBlock {
    reduce: Conv,
    v_long: Conv,
    v_mid: Conv,
    v_short: Conv,
    h_long: Conv,
    h_mid: Conv,
    h_short: Conv,
    s_long: Conv,
    s_mid: Conv,
    s_short: Conv,
    conv_final: ConvBn,
}

impl IntraclassBlock {
    fn new(w: &W, prefix: &str) -> Result<Self, Error> {
        let c =
            |name: &str| -> Result<Conv, Error> { Conv::new(w, &format!("{prefix}.{name}"), true) };
        Ok(Self {
            reduce: c("conv_reduce_channel")?,
            v_long: c("vertical_long_to_small_conv_longratio")?,
            v_mid: c("vertical_long_to_small_conv_midratio")?,
            v_short: c("vertical_long_to_small_conv_shortratio")?,
            h_long: c("horizontal_small_to_long_conv_longratio")?,
            h_mid: c("horizontal_small_to_long_conv_midratio")?,
            h_short: c("horizontal_small_to_long_conv_shortratio")?,
            s_long: c("symmetric_conv_long_longratio")?,
            s_mid: c("symmetric_conv_long_midratio")?,
            s_short: c("symmetric_conv_long_shortratio")?,
            conv_final: ConvBn::new(w, &format!("{prefix}.conv_final"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = self.reduce.forward(x, (1, 1))?;
        let y = ((self.s_long.forward(&y, (1, 1))? + self.v_long.forward(&y, (1, 1))?)?
            + self.h_long.forward(&y, (1, 1))?)?;
        let y = ((self.s_mid.forward(&y, (1, 1))? + self.v_mid.forward(&y, (1, 1))?)?
            + self.h_mid.forward(&y, (1, 1))?)?;
        let y = ((self.s_short.forward(&y, (1, 1))? + self.v_short.forward(&y, (1, 1))?)?
            + self.h_short.forward(&y, (1, 1))?)?;
        // conv_final is a ConvBatchnormLayer with relu activation (default).
        let y = self.conv_final.forward(&y)?.relu()?;
        Ok((x + y)?)
    }
}

/// Medium-tier detection neck.
pub struct MediumDetNeck {
    channel_adjust: Vec<Conv>,
    feature_proj: Vec<Conv>,
    pan_head: Vec<Conv>,
    pan_lateral: Vec<Conv>,
    intraclass: Vec<IntraclassBlock>,
    head: DbHead,
    scale_factor_list: Vec<usize>,
}

impl MediumDetNeck {
    pub fn new(w: &W, cfg: &MediumDetConfig) -> Result<Self, Error> {
        let n = 4;
        let mut channel_adjust = Vec::with_capacity(n);
        let mut feature_proj = Vec::with_capacity(n);
        let mut pan_head = Vec::with_capacity(n - 1);
        let mut pan_lateral = Vec::with_capacity(n);
        for i in 0..n {
            channel_adjust.push(Conv::new(
                w,
                &format!("model.neck.input_channel_adjustment_convolution.{i}"),
                false,
            )?);
            feature_proj.push(Conv::new(
                w,
                &format!("model.neck.input_feature_projection_convolution.{i}"),
                true,
            )?);
            if i > 0 {
                pan_head.push(Conv::new(
                    w,
                    &format!("model.neck.path_aggregation_head_convolution.{}", i - 1),
                    false,
                )?);
            }
            pan_lateral.push(Conv::new(
                w,
                &format!("model.neck.path_aggregation_lateral_convolution.{i}"),
                true,
            )?);
        }
        let mut intraclass = Vec::with_capacity(cfg.intraclass_block_number);
        for i in 0..cfg.intraclass_block_number {
            intraclass.push(IntraclassBlock::new(
                w,
                &format!("model.neck.intraclass_blocks.{i}"),
            )?);
        }
        let head = DbHead::new(w)?;
        Ok(Self {
            channel_adjust,
            feature_proj,
            pan_head,
            pan_lateral,
            intraclass,
            head,
            scale_factor_list: cfg.scale_factor_list.clone(),
        })
    }

    pub fn head(&self) -> &DbHead {
        &self.head
    }

    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, stages: &[Tensor]) -> Result<Tensor, Error> {
        let n = stages.len();
        // 1. channel adjustment (1×1 → neck_out)
        let mut adj = Vec::with_capacity(n);
        for i in 0..n {
            adj.push(self.channel_adjust[i].forward(&stages[i], (1, 1))?);
        }
        // 2. top-down FPN
        let mut top_down: Vec<Option<Tensor>> = vec![None; n];
        top_down[n - 1] = Some(adj[n - 1].clone());
        for i in (0..n - 1).rev() {
            let t = (adj[i].clone() + up2(top_down[i + 1].as_ref().unwrap())?)?;
            top_down[i] = Some(t);
        }
        // 3. 9×9 projection → neck_out/4
        let mut projected = Vec::with_capacity(n);
        for i in 0..n {
            projected.push(self.feature_proj[i].forward(top_down[i].as_ref().unwrap(), (1, 1))?);
        }
        // 4. bottom-up (PAN) with stride-2 3×3 convs
        let mut bottom_up = Vec::with_capacity(n);
        bottom_up.push(projected[0].clone());
        for i in 1..n {
            let t = (projected[i].clone()
                + self.pan_head[i - 1].forward(&bottom_up[i - 1], (2, 2))?)?;
            bottom_up.push(t);
        }
        // 5. lateral refinement (9×9)
        let mut lateral = Vec::with_capacity(n);
        for i in 0..n {
            let input = if i == 0 {
                projected[0].clone()
            } else {
                bottom_up[i].clone()
            };
            lateral.push(self.pan_lateral[i].forward(&input, (1, 1))?);
        }
        // 6. intra-class blocks
        let mut intra = Vec::with_capacity(n);
        for i in 0..self.intraclass.len() {
            intra.push(self.intraclass[i].forward(&lateral[i])?);
        }
        // 7. upsample to the largest scale and concatenate in reversed order.
        let (_, _, th, tw) = intra[0].dims4()?;
        let mut upsampled = Vec::with_capacity(n);
        for i in 0..self.intraclass.len() {
            let scale = self.scale_factor_list.get(i).copied().unwrap_or(1);
            let f = if scale > 1 {
                intra[i].upsample_nearest2d(th, tw)?
            } else {
                intra[i].clone()
            };
            upsampled.push(f);
        }
        // reversed: [intra[3], intra[2], intra[1], intra[0]]
        let mut reversed = upsampled;
        reversed.reverse();
        Ok(Tensor::cat(&reversed, 1)?)
    }
}

/// Nearest-neighbor 2× upsampling.
fn up2(x: &Tensor) -> Result<Tensor, Error> {
    let (_, _, h, w) = x.dims4()?;
    Ok(x.upsample_nearest2d(h * 2, w * 2)?)
}
