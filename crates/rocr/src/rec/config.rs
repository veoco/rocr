//! Parsing of the recognition model `config.json`.

use serde_json::Value;

use crate::error::Error;

/// Configuration of one LCNetV4 block.
#[derive(Debug, Clone)]
pub struct BlockConfig {
    /// Depthwise kernel size (always 3 in PP-OCRv6).
    pub kernel: usize,
    pub in_ch: usize,
    pub out_ch: usize,
    /// Convolution stride `(h, w)`.
    pub stride: (usize, usize),
    /// Whether a squeeze-and-excitation module is used.
    pub use_se: bool,
}

/// LCNetV4 backbone configuration.
#[derive(Debug, Clone)]
pub struct BackboneConfig {
    /// Stem channels `(in, mid, out)` (e.g. `(3, 48, 96)`).
    pub stem_channels: (usize, usize, usize),
    /// Stem variant: `"large"` (stem1..4) or `"small"` (conv1 + conv2).
    pub stem_type: String,
    /// Per-stage block configurations.
    pub stages: Vec<Vec<BlockConfig>>,
}

/// Full recognition model configuration.
#[derive(Debug, Clone)]
pub struct RecConfig {
    pub backbone: BackboneConfig,
    pub hidden_size: usize,
    pub depth: usize,
    #[allow(dead_code)]
    pub head_out_channels: usize,
}

fn parse_stride(v: &Value) -> Result<(usize, usize), Error> {
    if v.is_array() {
        let a = v
            .as_array()
            .ok_or_else(|| Error::Config("stride not array".into()))?;
        Ok((
            a[0].as_u64()
                .ok_or_else(|| Error::Config("stride[0] not int".into()))? as usize,
            a[1].as_u64()
                .ok_or_else(|| Error::Config("stride[1] not int".into()))? as usize,
        ))
    } else {
        let s = v
            .as_u64()
            .ok_or_else(|| Error::Config("stride not int".into()))? as usize;
        Ok((s, s))
    }
}

fn parse_block(v: &Value) -> Result<BlockConfig, Error> {
    let a = v
        .as_array()
        .ok_or_else(|| Error::Config("block not array".into()))?;
    if a.len() < 5 {
        return Err(Error::Config("block config has <5 fields".into()));
    }
    Ok(BlockConfig {
        kernel: a[0].as_u64().unwrap_or(3) as usize,
        in_ch: a[1].as_u64().unwrap_or(0) as usize,
        out_ch: a[2].as_u64().unwrap_or(0) as usize,
        stride: parse_stride(&a[3])?,
        use_se: a[4].as_bool().unwrap_or(false),
    })
}

impl BackboneConfig {
    /// Parse from a `backbone_config` JSON object (shared by det and rec).
    pub fn from_json(bb: &Value) -> Result<Self, Error> {
        let sc = bb
            .get("stem_channels")
            .and_then(|x| x.as_array())
            .ok_or_else(|| Error::Config("missing stem_channels".into()))?;
        let stem_channels = (
            sc[0].as_u64().unwrap_or(3) as usize,
            sc[1].as_u64().unwrap_or(48) as usize,
            sc[2].as_u64().unwrap_or(96) as usize,
        );
        let stages_json = bb
            .get("block_configs")
            .and_then(|x| x.as_array())
            .ok_or_else(|| Error::Config("missing block_configs".into()))?;
        let stages = stages_json
            .iter()
            .map(|stage| {
                stage
                    .as_array()
                    .ok_or_else(|| Error::Config("stage not array".into()))?
                    .iter()
                    .map(parse_block)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let stem_type = bb
            .get("stem_type")
            .and_then(|x| x.as_str())
            .unwrap_or("large")
            .to_string();
        Ok(Self {
            stem_channels,
            stem_type,
            stages,
        })
    }
}

impl RecConfig {
    pub fn from_json(v: &Value) -> Result<Self, Error> {
        let bb = &v["backbone_config"];
        let backbone = BackboneConfig::from_json(bb)?;
        let hidden_size = v.get("hidden_size").and_then(|x| x.as_u64()).unwrap_or(120) as usize;
        let depth = v.get("depth").and_then(|x| x.as_u64()).unwrap_or(2) as usize;
        let head_out_channels = v
            .get("head_out_channels")
            .and_then(|x| x.as_u64())
            .unwrap_or(18710) as usize;
        Ok(Self {
            backbone,
            hidden_size,
            depth,
            head_out_channels,
        })
    }
}
