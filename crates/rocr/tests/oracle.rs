//! Oracle tests: compare rocr's candle output / preprocessing against numeric
//! fixtures produced from the official PP-OCRv6 ONNX models by
//! `scripts/verify_onnx.py`.
//!
//! These tests are skipped when the fixtures are absent (e.g. fresh clone or
//! CI without dev-models).

use std::path::{Path, PathBuf};

use candle_core::Tensor;
use rocr::common::preprocess::{det_preprocess, max_abs_diff, rec_preprocess};
use rocr::det::DetModel;
use rocr::rec::RecModel;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn ref_dir() -> PathBuf {
    workspace_root().join("dev-models").join("reference")
}

fn assets_dir() -> PathBuf {
    workspace_root().join("assets")
}

fn load_npy(name: &str) -> Option<Tensor> {
    let path = ref_dir().join(name);
    if !path.exists() {
        return None;
    }
    Some(Tensor::read_npy(path).expect("read npy fixture"))
}

#[test]
fn rec_preprocessing_matches_oracle() {
    let Some(fixture_input) = load_npy("rec_input_small.npy") else {
        eprintln!("skip: fixture rec_input_small.npy not present");
        return;
    };
    let img = image::open(assets_dir().join("line_long.png")).unwrap();
    let (tensor, tw) = rec_preprocess(&img, &candle_core::Device::Cpu).unwrap();
    assert_eq!(
        tensor.dims(),
        fixture_input.dims(),
        "shape mismatch (w={tw})"
    );
    let diff = max_abs_diff(&tensor, &fixture_input).unwrap();
    // Resize interpolation (cv2 vs ours) causes small per-pixel differences.
    assert!(
        diff < 0.02,
        "rec preprocess deviates too much from oracle: max|diff|={diff}"
    );
}

// Note: the official `*_det_onnx` export uses a different checkpoint from the
// safetensors model, so the ONNX is not usable as a det oracle. Instead the
// reference is produced by the PaddleOCR transformers engine (same safetensors
// weights) via `scripts/verify_paddle_det.py`; see `paddle_det_*` fixtures.

#[test]
fn det_network_matches_paddle() {
    let base = workspace_root().join("dev-models").join("reference");
    let input = base.join("paddle_det_input.npy");
    let output = base.join("paddle_det_last_hidden_state.npy");
    if !input.exists() || !output.exists() {
        eprintln!("skip: paddle det reference not present");
        return;
    }
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-OCRv6_small_det_safetensors");
    if !repo.exists() {
        eprintln!("skip: det model repo not present");
        return;
    }
    let model = DetModel::new(&repo, &candle_core::Device::Cpu).unwrap();
    let x = Tensor::read_npy(&input).unwrap();
    let out = model.forward(&x).unwrap();
    let ref_out = Tensor::read_npy(&output).unwrap();
    assert_eq!(out.dims(), ref_out.dims(), "output shape mismatch");
    let diff = max_abs_diff(&out, &ref_out).unwrap();
    assert!(
        diff < 0.02,
        "det network deviates from paddle transformers reference: max|diff|={diff}"
    );
}

// The tiny rec tier is sensitive to the exact resize interpolation, so its
// reference fixture is produced by the PaddleOCR transformers engine (same
// weights) rather than by the ONNX/cv2 pipeline.
#[test]
fn tiny_rec_matches_paddle() {
    let base = workspace_root().join("dev-models").join("reference");
    let input = base.join("paddle_tiny_rec_input.npy");
    let output = base.join("paddle_tiny_rec_output.npy");
    if !input.exists() || !output.exists() {
        eprintln!("skip: paddle tiny rec reference not present");
        return;
    }
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-OCRv6_tiny_rec_safetensors");
    if !repo.exists() {
        eprintln!("skip: tiny rec model repo not present");
        return;
    }
    let model = RecModel::new(&repo, &candle_core::Device::Cpu).unwrap();
    let x = Tensor::read_npy(&input).unwrap();
    let out = model.forward(&x).unwrap();
    let ref_out = Tensor::read_npy(&output).unwrap();
    assert_eq!(out.dims(), ref_out.dims(), "output shape mismatch");
    let diff = max_abs_diff(&out, &ref_out).unwrap();
    assert!(
        diff < 1e-3,
        "tiny rec deviates from paddle transformers reference: max|diff|={diff}"
    );
}

#[test]
fn rec_network_matches_oracle() {
    let Some(input) = load_npy("rec_input_small.npy") else {
        eprintln!("skip: rec fixture not present");
        return;
    };
    let Some(logits) = load_npy("rec_logits_small.npy") else {
        eprintln!("skip: rec logits fixture not present");
        return;
    };
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-OCRv6_small_rec_safetensors");
    if !repo.exists() {
        eprintln!("skip: rec model repo not present");
        return;
    }
    let model = RecModel::new(&repo, &candle_core::Device::Cpu).unwrap();
    let out = model.forward(&input).unwrap();
    assert_eq!(out.dims(), logits.dims(), "output shape mismatch");
    let diff = max_abs_diff(&out, &logits).unwrap();
    assert!(
        diff < 1e-3,
        "rec network deviates too much from ONNX oracle: max|diff|={diff}"
    );
}

#[test]
fn full_pipeline_ocr_document() {
    let base = workspace_root().join("dev-models");
    let det_repo = base.join("PP-OCRv6_small_det_safetensors");
    let rec_repo = base.join("PP-OCRv6_small_rec_safetensors");
    if !det_repo.exists() || !rec_repo.exists() {
        eprintln!("skip: models not present");
        return;
    }
    use rocr::{DeviceKind, ModelTier, Ocr, OcrConfig};
    let ocr = Ocr::new(OcrConfig {
        model_tier: ModelTier::Small,
        device: DeviceKind::Cpu,
        model_dir: base,
        ..Default::default()
    })
    .unwrap();
    let img = image::open(assets_dir().join("doc.png")).unwrap();
    let results = ocr.recognize(&img).unwrap();
    eprintln!("pipeline: {} text lines recognized", results.len());
    for r in &results {
        eprintln!("  {:.4}  {}", r.confidence, r.text);
    }
    // Smoke test: the pipeline must run and return a result. Full end-to-end
    // accuracy is currently limited by the detection model (see README /
    // KNOWN-ISSUES).
    assert!(!results.is_empty(), "no text recognized");
}

#[test]
fn rec_recognizes_text_lines() {
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-OCRv6_small_rec_safetensors");
    if !repo.exists() {
        eprintln!("skip: rec model repo not present");
        return;
    }
    let model = RecModel::new(&repo, &candle_core::Device::Cpu).unwrap();
    let cases = [
        ("line_en.png", "Hello World OCR"),
        ("line_long.png", "hello world hello world hello"),
        ("line_cn.png", "你好世界"),
    ];
    for (file, expected) in cases {
        let img = image::open(assets_dir().join(file)).unwrap();
        let (text, conf) = model.recognize(&img).unwrap();
        eprintln!("recognize {file}: {text:?} conf={conf:.4}");
        assert!(
            text == expected || text.contains(expected) || expected.contains(&text),
            "mismatch: got {text:?}, want {expected:?}"
        );
        assert!(conf > 0.5, "low confidence {conf}");
    }
}

// Text-line orientation: fixture produced by the transformers engine (same
// safetensors weights) via `scripts/verify_textline_ori.py`.
#[test]
fn textline_orient_matches_paddle() {
    let base = workspace_root().join("dev-models").join("reference");
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-LCNet_x0_25_textline_ori_safetensors");
    if !repo.exists() {
        eprintln!("skip: orient model repo not present");
        return;
    }
    let model = rocr::PpLcNet::new(&repo, &candle_core::Device::Cpu).unwrap();
    for (tag, tolerance) in [("cn", 1e-2f32), ("cn_rot", 1e-2f32), ("en", 1e-2f32)] {
        let input = base.join(format!("paddle_orient_input_{tag}.npy"));
        let output = base.join(format!("paddle_orient_output_{tag}.npy"));
        if !input.exists() || !output.exists() {
            eprintln!("skip: orient fixture {tag} not present");
            continue;
        }
        let x = Tensor::read_npy(&input).unwrap();
        let out = model.forward(&x).unwrap();
        let ref_out = Tensor::read_npy(&output).unwrap();
        assert_eq!(out.dims(), ref_out.dims(), "{tag}: output shape mismatch");
        let diff = max_abs_diff(&out, &ref_out).unwrap();
        eprintln!("orient {tag} logits diff = {diff}");
        assert!(
            diff < tolerance,
            "{tag}: orient deviates from transformers reference: max|diff|={diff}"
        );
        // The argmax must agree (0 → upright, 1 → rotated 180°).
        let got = out.to_vec2::<f32>().unwrap()[0].clone();
        let want = ref_out.to_vec2::<f32>().unwrap()[0].clone();
        let (gi, wi) = (
            got.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            want.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
        );
        assert_eq!(gi, wi, "{tag}: argmax mismatch");
    }
}

// The orientation classifier should rotate an upside-down crop and keep an
// upright one as-is.
#[test]
fn textline_orient_rotates_crops() {
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-LCNet_x0_25_textline_ori_safetensors");
    if !repo.exists() {
        eprintln!("skip: orient model repo not present");
        return;
    }
    let model = rocr::PpLcNet::new(&repo, &candle_core::Device::Cpu).unwrap();
    let img = image::open(assets_dir().join("line_cn.png")).unwrap();
    assert!(
        !rocr::orient::classify_textline(&model, &img).unwrap(),
        "upright crop misclassified"
    );
    let rot = img.rotate180();
    assert!(
        rocr::orient::classify_textline(&model, &rot).unwrap(),
        "upside-down crop not detected"
    );
}

// Document orientation: fixture produced by the transformers engine via
// `scripts/verify_doc_ori.py`.
#[test]
fn doc_orient_matches_paddle() {
    let base = workspace_root().join("dev-models").join("reference");
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-LCNet_x1_0_doc_ori_safetensors");
    if !repo.exists() {
        eprintln!("skip: doc orient model repo not present");
        return;
    }
    let model = rocr::PpLcNet::new(&repo, &candle_core::Device::Cpu).unwrap();
    for tag in ["orig", "rot90", "rot180", "rot270"] {
        let input = base.join(format!("paddle_docori_input_{tag}.npy"));
        let output = base.join(format!("paddle_docori_output_{tag}.npy"));
        if !input.exists() || !output.exists() {
            eprintln!("skip: doc orient fixture {tag} not present");
            continue;
        }
        let x = Tensor::read_npy(&input).unwrap();
        let out = model.forward(&x).unwrap();
        let ref_out = Tensor::read_npy(&output).unwrap();
        assert_eq!(out.dims(), ref_out.dims(), "{tag}: output shape mismatch");
        let diff = max_abs_diff(&out, &ref_out).unwrap();
        eprintln!("doc orient {tag} logits diff = {diff}");
        assert!(
            diff < 1e-2,
            "{tag}: doc orient deviates from transformers reference: max|diff|={diff}"
        );
        let got = out.to_vec2::<f32>().unwrap()[0].clone();
        let want = ref_out.to_vec2::<f32>().unwrap()[0].clone();
        let (gi, wi) = (
            got.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
            want.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
        );
        assert_eq!(gi, wi, "{tag}: argmax mismatch");
    }
}

// The doc-orientation classifier must report the rotation of rotated pages.
#[test]
fn doc_orient_rotates_pages() {
    let repo = workspace_root()
        .join("dev-models")
        .join("PP-LCNet_x1_0_doc_ori_safetensors");
    if !repo.exists() {
        eprintln!("skip: doc orient model repo not present");
        return;
    }
    let model = rocr::PpLcNet::new(&repo, &candle_core::Device::Cpu).unwrap();
    let img = image::open(assets_dir().join("doc.png")).unwrap();
    assert_eq!(rocr::orient::classify_doc(&model, &img).unwrap(), 0);
    // image::rotate90 is clockwise, rotate270 is counter-clockwise.
    assert_eq!(
        rocr::orient::classify_doc(&model, &img.rotate270()).unwrap(),
        3
    );
    assert_eq!(
        rocr::orient::classify_doc(&model, &img.rotate180()).unwrap(),
        2
    );
    assert_eq!(
        rocr::orient::classify_doc(&model, &img.rotate90()).unwrap(),
        1
    );
}

#[test]
fn det_preprocessing_matches_oracle() {
    let Some(fixture_input) = load_npy("det_input_small.npy") else {
        eprintln!("skip: fixture det_input_small.npy not present");
        return;
    };
    let img = image::open(assets_dir().join("doc.png")).unwrap();
    let (tensor, nh, nw) = det_preprocess(&img, &candle_core::Device::Cpu).unwrap();
    assert_eq!(
        tensor.dims(),
        fixture_input.dims(),
        "shape mismatch ({nh}x{nw})"
    );
    let diff = max_abs_diff(&tensor, &fixture_input).unwrap();
    assert!(
        diff < 0.02,
        "det preprocess deviates too much from oracle: max|diff|={diff}"
    );
}

/// Levenshtein distance between two strings (as chars).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = if ca == cb {
                prev[j]
            } else {
                prev[j].min(prev[j + 1]).min(cur[j]) + 1
            };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Similarity in `[0, 1]`: `1 - dist / max_len` (empty-safe).
fn edit_similarity(a: &str, b: &str) -> f32 {
    let m = a.chars().count().max(b.chars().count());
    if m == 0 {
        return 1.0;
    }
    1.0 - edit_distance(a, b) as f32 / m as f32
}

// End-to-end golden check: rocr's pipeline output on a real page must largely
// match what PaddleOCR's own engine recognizes. The reference is generated by
// `scripts/verify_golden.py`; matching is fuzzy (edit distance) because the
// reference engine itself is imperfect on dense boarding-pass layouts.
#[test]
fn full_pipeline_matches_paddle_golden() {
    let golden = ref_dir().join("golden_small_general_ocr_002.txt");
    let img = assets_dir().join("general_ocr_002.png");
    if !golden.exists() || !img.exists() {
        eprintln!("skip: golden reference or image not present");
        return;
    }
    let ocr = rocr::Ocr::new(rocr::OcrConfig {
        model_tier: rocr::ModelTier::Small,
        device: rocr::DeviceKind::Cpu,
        model_dir: workspace_root().join("dev-models"),
        ..Default::default()
    })
    .unwrap();
    let results = ocr.recognize(&image::open(&img).unwrap()).unwrap();
    let golden_lines: Vec<String> = std::fs::read_to_string(&golden)
        .unwrap()
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut matched = 0;
    for r in &results {
        let best = golden_lines
            .iter()
            .map(|g| edit_similarity(&r.text, g))
            .fold(0.0f32, f32::max);
        if best >= 0.5 {
            matched += 1;
        }
        eprintln!("  {best:.2}  {}", r.text);
    }
    let rate = matched as f32 / results.len().max(1) as f32;
    eprintln!(
        "golden: matched {matched}/{} lines (rate {rate:.2}) vs {} golden lines",
        results.len(),
        golden_lines.len()
    );
    assert!(
        rate >= 0.6,
        "only {rate:.2} of rocr lines match the paddle golden reference"
    );
}
