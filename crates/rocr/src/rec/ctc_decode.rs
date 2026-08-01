//! Greedy CTC decoding.

use candle_core::Tensor;

use crate::error::Error;

/// A CTC-decoded token sequence with per-token confidence.
#[derive(Debug, Clone, Default)]
pub struct CtcDecoded {
    /// Class indices (blank=0 excluded).
    pub ids: Vec<usize>,
    /// Max probability of each decoded token.
    pub confidences: Vec<f32>,
}

fn argmax(row: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    (best, best_v)
}

/// Greedy CTC decode on a `[T, C]` probability tensor (blank = index 0).
/// Collapses consecutive repeats and removes blanks.
pub fn ctc_greedy(probs: &Tensor) -> Result<CtcDecoded, Error> {
    let data = probs.to_vec2::<f32>()?;
    let mut ids = Vec::new();
    let mut confidences = Vec::new();
    let mut prev = 0usize; // blank
    for row in &data {
        let (idx, max_v) = argmax(row);
        if idx != prev && idx != 0 {
            ids.push(idx);
            confidences.push(max_v);
        }
        prev = idx;
    }
    Ok(CtcDecoded { ids, confidences })
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    fn probs(v: &[&[f32]]) -> Tensor {
        Tensor::from_vec(v.concat(), (v.len(), v[0].len()), &Device::Cpu).unwrap()
    }

    #[test]
    fn collapses_repeats_and_blanks() {
        // T=8, C=4. blank=0.
        let p = probs(&[
            &[0.9, 0.1, 0.0, 0.0], // blank
            &[0.1, 0.9, 0.0, 0.0], // 'a'
            &[0.1, 0.9, 0.0, 0.0], // 'a' repeat → collapse
            &[0.9, 0.0, 0.1, 0.0], // blank
            &[0.1, 0.0, 0.9, 0.0], // 'b'
            &[0.1, 0.0, 0.0, 0.9], // 'c'
            &[0.9, 0.1, 0.0, 0.0], // blank
            &[0.1, 0.0, 0.9, 0.0], // 'b'
        ]);
        let out = ctc_greedy(&p).unwrap();
        assert_eq!(out.ids, vec![1, 2, 3, 2]);
        assert!(out.confidences.iter().all(|&c| (c - 0.9).abs() < 1e-5));
    }

    #[test]
    fn all_blank_is_empty() {
        let p = probs(&[&[0.8, 0.2, 0.0], &[0.9, 0.1, 0.0]]);
        assert!(ctc_greedy(&p).unwrap().ids.is_empty());
    }

    #[test]
    fn single_step_single_char() {
        let p = probs(&[&[0.1, 0.9, 0.0]]);
        let out = ctc_greedy(&p).unwrap();
        assert_eq!(out.ids, vec![1]);
        assert_eq!(out.confidences, vec![0.9]);
    }

    #[test]
    fn leading_and_trailing_blanks_are_dropped() {
        let p = probs(&[
            &[0.9, 0.1, 0.0], // blank
            &[0.1, 0.9, 0.0], // 'a'
            &[0.9, 0.1, 0.0], // blank
        ]);
        assert_eq!(ctc_greedy(&p).unwrap().ids, vec![1]);
    }

    #[test]
    fn repeats_split_by_blank_are_kept() {
        // aa _ bb _ aa  → [a, b, a] (blank separates the repeats).
        let p = probs(&[
            &[0.1, 0.9, 0.0],
            &[0.1, 0.9, 0.0],
            &[0.9, 0.1, 0.0],
            &[0.1, 0.0, 0.9],
            &[0.1, 0.0, 0.9],
            &[0.9, 0.1, 0.0],
            &[0.1, 0.9, 0.0],
            &[0.1, 0.9, 0.0],
        ]);
        assert_eq!(ctc_greedy(&p).unwrap().ids, vec![1, 2, 1]);
    }

    #[test]
    fn adjacent_repeats_collapse() {
        let p = probs(&[&[0.1, 0.9, 0.0], &[0.1, 0.9, 0.0], &[0.1, 0.9, 0.0]]);
        assert_eq!(ctc_greedy(&p).unwrap().ids, vec![1]);
    }

    #[test]
    fn empty_sequence_is_empty() {
        let p = Tensor::from_vec(Vec::<f32>::new(), (0, 3), &Device::Cpu).unwrap();
        assert!(ctc_greedy(&p).unwrap().ids.is_empty());
    }
}
