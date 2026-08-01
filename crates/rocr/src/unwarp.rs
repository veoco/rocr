//! Document unwarping / rectification (optional, default off).
//!
//! **Not implemented.** PaddleOCR's unwarping module is the UVDoc model: it
//! takes a page image (e.g. `[1,3,712,488]`) and outputs a sampling grid
//! (`[1,2,45,31]`) used with `grid_sample`. No official safetensors +
//! `config.json` transformers checkpoint exists (only MLX/ExecuTorch
//! conversions), and the network is a deep encoder–decoder, so it cannot be
//! loaded by rocr's weight-driven loader. The `enable_doc_unwarping` config
//! flag stays a placeholder; enabling it has no effect.
//!
//! Document orientation classification, by contrast, IS implemented
//! (`PP-LCNet_x1_0_doc_ori` via [`crate::orient`]).
