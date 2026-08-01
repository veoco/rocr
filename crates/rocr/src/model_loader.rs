//! Loading and parsing of locally stored PP-OCRv6 model files.
//!
//! Each model repository (e.g. `PP-OCRv6_small_rec_safetensors`) contains:
//! - `model.safetensors`        — the weights
//! - `config.json`              — architecture parameters
//! - `preprocessor_config.json` — preprocessing parameters (+ embedded
//!   character list for recognition)
//! - `inference.yml`            — official inference graph (reference only)
//!
//! The library never downloads models; callers provide the directory.

use crate::error::Error;
use std::path::{Path, PathBuf};

/// Names of the files expected inside a model repository directory.
pub const CONFIG_FILE: &str = "config.json";
pub const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// Resolve the directory of a model repository under a base model dir.
pub fn repo_dir(base: &Path, repo_suffix: &str) -> PathBuf {
    base.join(format!("PP-OCRv6_{repo_suffix}_safetensors"))
}

/// Read the raw bytes of a file from a model repo, with a friendly error.
pub fn read_repo_file(repo: &Path, name: &str) -> Result<Vec<u8>, Error> {
    let path = repo.join(name);
    if !path.exists() {
        return Err(Error::ModelFileMissing(path.display().to_string()));
    }
    Ok(std::fs::read(&path)?)
}

/// Parse a `config.json` / `preprocessor_config.json` into a JSON value.
pub fn load_json(repo: &Path, name: &str) -> Result<serde_json::Value, Error> {
    let bytes = read_repo_file(repo, name)?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Config(format!("{name}: {e}")))
}

/// Build a [`candle_nn::VarBuilder`] over a model repo's safetensors weights.
pub fn load_weights<'a>(
    repo: &'a Path,
    dtype: candle_core::DType,
    device: &'a candle_core::Device,
) -> Result<candle_nn::VarBuilder<'a>, Error> {
    let tensors = load_tensors(repo, device)?;
    Ok(candle_nn::VarBuilder::from_tensors(tensors, dtype, device))
}

/// Load all safetensors weights into a name → tensor map.
pub fn load_tensors(
    repo: &Path,
    device: &candle_core::Device,
) -> Result<std::collections::HashMap<String, candle_core::Tensor>, Error> {
    let path = repo.join(WEIGHTS_FILE);
    if !path.exists() {
        return Err(Error::ModelFileMissing(path.display().to_string()));
    }
    candle_core::safetensors::load(&path, device)
        .map_err(|e| Error::Safetensors(format!("{}: {e}", path.display())))
}
