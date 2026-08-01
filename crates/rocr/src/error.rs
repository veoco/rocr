//! Error types for rocr.

use std::path::PathBuf;

/// Errors returned by the rocr library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A feature has not been implemented yet (used during development).
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    /// The provided model directory does not exist.
    #[error("model directory does not exist: {0}")]
    ModelDirMissing(PathBuf),

    /// A required model file is missing from the model directory.
    #[error("required model file not found: {0}")]
    ModelFileMissing(String),

    /// Failed to read / parse a model configuration file.
    #[error("config error: {0}")]
    Config(String),

    /// Failed to load safetensors weights.
    #[error("safetensors error: {0}")]
    Safetensors(String),

    /// Failed to decode an image.
    #[error("image error: {0}")]
    Image(String),

    /// The requested inference backend is unavailable (feature not enabled).
    #[error("unsupported backend: {0}")]
    UnsupportedBackend(String),

    /// Underlying candle (tensor) error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Underlying I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
