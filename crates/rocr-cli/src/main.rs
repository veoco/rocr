//! rocr-cli — command-line OCR tool built on the `rocr` library.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rocr::{DeviceKind, ModelTier, Ocr, OcrConfig};

#[derive(Parser, Debug)]
#[command(name = "rocr", version, about = "Pure Rust OCR based on PP-OCRv6")]
struct Args {
    /// Input image file(s). May be given multiple times.
    #[arg(short, long)]
    image: Vec<PathBuf>,

    /// Model size tier.
    #[arg(long, value_enum, default_value_t = TierArg::Small)]
    model: TierArg,

    /// Inference device.
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,

    /// Directory containing the manually downloaded model repositories.
    #[arg(long)]
    model_dir: PathBuf,

    /// Enable document orientation classification (0/90/180/270°), default off.
    #[arg(long)]
    doc_orientation: bool,

    /// Disable text-line orientation classification (on by default).
    #[arg(long)]
    no_textline_orientation: bool,

    /// Print per-image timing information to stderr.
    #[arg(long)]
    verbose: bool,

    /// Output results as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TierArg {
    Tiny,
    Small,
    Medium,
}

impl From<TierArg> for ModelTier {
    fn from(v: TierArg) -> Self {
        match v {
            TierArg::Tiny => ModelTier::Tiny,
            TierArg::Small => ModelTier::Small,
            TierArg::Medium => ModelTier::Medium,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DeviceArg {
    Cpu,
    Cuda,
    Metal,
}

impl From<DeviceArg> for DeviceKind {
    fn from(v: DeviceArg) -> Self {
        match v {
            DeviceArg::Cpu => DeviceKind::Cpu,
            DeviceArg::Cuda => DeviceKind::Cuda,
            DeviceArg::Metal => DeviceKind::Metal,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.image.is_empty() {
        anyhow::bail!("at least one --image is required");
    }

    let config = OcrConfig {
        model_tier: args.model.into(),
        device: args.device.into(),
        model_dir: args.model_dir,
        enable_doc_orientation: args.doc_orientation,
        enable_textline_orientation: !args.no_textline_orientation,
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let ocr = Ocr::new(config)?;
    if args.verbose {
        eprintln!("models loaded in {:?}", t0.elapsed());
    }

    for path in &args.image {
        let t = std::time::Instant::now();
        let img = image::open(path)?;
        let results = ocr.recognize(&img)?;
        if args.json {
            let out = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "text": r.text,
                        "confidence": r.confidence,
                        "polygon": r.polygon,
                    })
                })
                .collect::<Vec<_>>();
            if args.image.len() > 1 {
                // Emit a JSON object keyed by filename when batching.
                println!("{}", serde_json::json!({ path.display().to_string(): out }));
            } else {
                println!("{}", serde_json::to_string_pretty(&out)?);
            }
        } else {
            for r in &results {
                println!("{}\t{:.4}\t{:?}", r.text, r.confidence, r.polygon);
            }
        }
        if args.verbose {
            eprintln!(
                "{}: {} lines in {:?}",
                path.display(),
                results.len(),
                t.elapsed()
            );
        }
    }
    Ok(())
}
