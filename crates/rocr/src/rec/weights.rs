//! Safetensors weight access helper.

use std::collections::HashMap;

use candle_core::Tensor;

use crate::error::Error;

/// Read-only accessor over a name → tensor weight map.
#[derive(Clone, Copy)]
pub struct W<'a>(pub &'a HashMap<String, Tensor>);

impl<'a> W<'a> {
    pub fn get(&self, key: &str) -> Result<Tensor, Error> {
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| Error::ModelFileMissing(format!("missing weight: {key}")))
    }

    pub fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
}
