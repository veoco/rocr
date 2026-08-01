//! RepLKFPN detection neck + DB head.
//!
//! Structure (verified against the official ONNX graph):
//! ```text
//! ins[i] = insert_conv[i](stage[i])            # 1×1 → 96ch, self-gated SE
//! # top-down FPN with 2× nearest upsampling
//! f3 = ins[3]; f2 = ins[2] + up2(f3); f1 = ins[1] + up2(f2); f0 = ins[0] + up2(f1)
//! out[i] = input_conv[i](f[3-i])              # 7×7 DW + 1×1, self-gated SE, → 24ch
//! feat = cat(up8(out0), up4(out1), up2(out2), out3)          # 96ch at largest scale
//! prob = sigmoid(up2(up2(relu(conv3x3(feat)))))              # DB head
//! ```

use candle_core::Tensor;
use candle_nn::{BatchNorm, ModuleT};

use crate::error::Error;
use crate::rec::backbone::conv2d_2d;
use crate::rec::weights::W;

const BN_EPS: f64 = 1e-5;
/// Number of FPN levels (4 backbone stages).
const LEVELS: usize = 4;

/// `y = x + x * SE(x)` — self-gated squeeze-and-excitation residual.
///
/// The neck's SE gate uses `HardSigmoid(alpha=0.2, beta=0.5)`, i.e.
/// `clamp(0.2*x + 0.5, 0, 1)` — different from the backbone SE (alpha=1/6).
struct SeGate {
    conv1_w: Tensor,
    conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
}

impl SeGate {
    fn new(w: &W, prefix: &str) -> Result<Self, Error> {
        Ok(Self {
            conv1_w: w.get(&format!("{prefix}.conv1.weight"))?,
            conv1_b: w.get(&format!("{prefix}.conv1.bias"))?,
            conv2_w: w.get(&format!("{prefix}.conv2.weight"))?,
            conv2_b: w.get(&format!("{prefix}.conv2.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (_, gated) = self.steps(x)?;
        Ok((x + gated)?)
    }

    /// HardSigmoid with alpha=0.2, beta=0.5: `clamp(0.2*x + 0.5, 0, 1)`.
    fn gate_activation(x: &Tensor) -> Result<Tensor, Error> {
        Ok((x.affine(0.2, 0.5)?.clamp(0f32, 1f32))?)
    }

    fn steps(&self, x: &Tensor) -> Result<(Tensor, Tensor), Error> {
        let s = x.mean_keepdim([2, 3])?;
        let s = conv2d_2d(&s, &self.conv1_w, Some(&self.conv1_b), (0, 0), (1, 1), 1)?.relu()?;
        let s = conv2d_2d(&s, &self.conv2_w, Some(&self.conv2_b), (0, 0), (1, 1), 1)?;
        let s = Self::gate_activation(&s)?;
        let gated = x.broadcast_mul(&s)?;
        Ok((s, gated))
    }
}

/// `insert_conv.{i}`: 1×1 projection + self-gated SE.
struct InsertConv {
    conv_w: Tensor,
    se: SeGate,
}

impl InsertConv {
    fn new(w: &W, i: usize) -> Result<Self, Error> {
        let base = format!("model.neck.insert_conv.{i}");
        Ok(Self {
            conv_w: w.get(&format!("{base}.in_conv.weight"))?,
            se: SeGate::new(w, &format!("{base}.squeeze_excitation_block"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let y = conv2d_2d(x, &self.conv_w, None, (0, 0), (1, 1), 1)?;
        self.se.forward(&y)
    }
}

/// `input_conv.{i}`: 7×7 depthwise conv + 1×1 projection + self-gated SE.
struct InputConv {
    dw_w: Tensor,
    dw_b: Tensor,
    pw_w: Tensor,
    se: SeGate,
}

impl InputConv {
    fn new(w: &W, i: usize) -> Result<Self, Error> {
        let base = format!("model.neck.input_conv.{i}");
        Ok(Self {
            dw_w: w.get(&format!("{base}.depthwise_convolution.weight"))?,
            dw_b: w.get(&format!("{base}.depthwise_convolution.bias"))?,
            pw_w: w.get(&format!("{base}.pointwise_convolution.weight"))?,
            se: SeGate::new(w, &format!("{base}.squeeze_excitation_module"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (c, _, kh, kw) = self.dw_w.dims4()?;
        let pad = (kh / 2, kw / 2);
        let y = conv2d_2d(x, &self.dw_w, Some(&self.dw_b), pad, (1, 1), c)?;
        let y = conv2d_2d(&y, &self.pw_w, None, (0, 0), (1, 1), 1)?;
        self.se.forward(&y)
    }
}

/// DB detection head: 3×3 conv + BN + relu, then two 2×2 stride-2 transposed
/// convolutions, then sigmoid → probability map at input resolution.
pub struct DbHead {
    conv_down_w: Tensor,
    conv_down_bn: BatchNorm,
    conv_up_w: Tensor,
    conv_up_b: Tensor,
    conv_up_bn: BatchNorm,
    conv_final_w: Tensor,
    conv_final_b: Tensor,
}

impl DbHead {
    pub(crate) fn new(w: &W) -> Result<Self, Error> {
        let cd = "head.conv_down";
        let cu = "head.conv_up";
        let cf = "head.conv_final";
        let conv_down_w = w.get(&format!("{cd}.convolution.weight"))?;
        let conv_up_w = w.get(&format!("{cu}.convolution.weight"))?;
        // conv_down is [out, in, kh, kw]; conv_up (transposed) is [in, out, kh, kw].
        let down_out = conv_down_w.dims()[0];
        let up_out = conv_up_w.dims()[1];
        let bn = |p: &str, c: usize| -> Result<BatchNorm, Error> {
            Ok(BatchNorm::new(
                c,
                w.get(&format!("{p}.norm.running_mean"))?,
                w.get(&format!("{p}.norm.running_var"))?,
                w.get(&format!("{p}.norm.weight"))?,
                w.get(&format!("{p}.norm.bias"))?,
                BN_EPS,
            )?)
        };
        Ok(Self {
            conv_down_w,
            conv_down_bn: bn(cd, down_out)?,
            conv_up_w,
            conv_up_b: w.get(&format!("{cu}.convolution.bias"))?,
            conv_up_bn: bn(cu, up_out)?,
            conv_final_w: w.get(&format!("{cf}.weight"))?,
            conv_final_b: w.get(&format!("{cf}.bias"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(candle_nn::ops::sigmoid(&self.forward_logits(x)?)?)
    }

    /// Head without the final sigmoid (raw logits).
    pub(crate) fn forward_logits(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (_, c_in, _, _) = self.conv_up_w.dims4()?;
        // conv_down: 3×3
        let mut x = x.conv2d(&self.conv_down_w, 1, 1, 1, 1)?;
        x = self.conv_down_bn.forward_t(&x, false)?.relu()?;
        // conv_up: 2×2 stride-2 transposed
        let x = x.conv_transpose2d(&self.conv_up_w, 0, 0, 2, 1)?;
        let x = x.broadcast_add(&self.conv_up_b.reshape((1, c_in, 1, 1))?)?;
        let x = self.conv_up_bn.forward_t(&x, false)?.relu()?;
        // conv_final: 2×2 stride-2 transposed → 1 channel.
        let x = x.conv_transpose2d(&self.conv_final_w, 0, 0, 2, 1)?;
        Ok(x.broadcast_add(&self.conv_final_b.reshape((1, 1, 1, 1))?)?)
    }
}

/// RepLKFPN neck.
pub struct RepLkFpn {
    insert_convs: Vec<InsertConv>,
    input_convs: Vec<InputConv>,
    head: DbHead,
}

impl RepLkFpn {
    pub fn new(w: &W) -> Result<Self, Error> {
        let mut insert_convs = Vec::with_capacity(LEVELS);
        let mut input_convs = Vec::with_capacity(LEVELS);
        for i in 0..LEVELS {
            insert_convs.push(InsertConv::new(w, i)?);
            input_convs.push(InputConv::new(w, i)?);
        }
        let head = DbHead::new(w)?;
        Ok(Self {
            insert_convs,
            input_convs,
            head,
        })
    }

    pub fn head(&self) -> &DbHead {
        &self.head
    }

    pub fn forward(&self, stages: &[Tensor]) -> Result<Tensor, Error> {
        // Project each stage to the neck channel (96).
        let mut ins = Vec::with_capacity(LEVELS);
        for (i, stage) in stages.iter().enumerate() {
            ins.push(self.insert_convs[i].forward(stage)?);
        }
        // Top-down FPN with 2× nearest upsampling.
        let f3 = ins[3].clone();
        let f2 = (ins[2].clone() + up2(&ins[3])?)?;
        let f1 = (ins[1].clone() + up2(&f2)?)?;
        let f0 = (ins[0].clone() + up2(&f1)?)?;
        let fused = [f0, f1, f2, f3]; // fused[0] is the largest scale (H/4)
                                      // input_conv[i] processes fused[i]; then upsample each to the largest
                                      // scale and concatenate in REVERSED order ([p5, p4, p3, p2]).
        let (_, _, th, tw) = fused[0].dims4()?;
        let p0 = self.input_convs[0].forward(&fused[0])?;
        let p1 = self.input_convs[1]
            .forward(&fused[1])?
            .upsample_nearest2d(th, tw)?;
        let p2 = self.input_convs[2]
            .forward(&fused[2])?
            .upsample_nearest2d(th, tw)?;
        let p3 = self.input_convs[3]
            .forward(&fused[3])?
            .upsample_nearest2d(th, tw)?;
        Ok(Tensor::cat(&[p3, p2, p1, p0], 1)?)
    }
}

/// Nearest-neighbor 2× upsampling.
fn up2(x: &Tensor) -> Result<Tensor, Error> {
    let (_, _, h, w) = x.dims4()?;
    Ok(x.upsample_nearest2d(h * 2, w * 2)?)
}
