//! # SnapOCR — Lightweight Desktop OCR Engine
//!
//! A fast, cross-platform OCR engine designed for desktop screenshot text
//! recognition, with native support for Chinese-English mixed text.
//!
//! ## Features
//! - **Pure Rust**: Uses `tract-onnx` for inference with zero C/C++ dependencies
//! - **Fast**: Sub-100ms recognition on modern CPUs for typical screenshots
//! - **Accurate**: Unified Chinese-English vocabulary eliminates mixed-text errors
//! - **Lightweight**: < 40MB RAM, < 25MB model files
//! - **Cross-platform**: Windows, macOS, Linux (x86_64, ARM64)
//!
//! ## Quick Start
//! ```no_run
//! use snap_ocr::{SnapOcrEngine, OcrEngineConfig};
//!
//! let config = OcrEngineConfig {
//!     model_dir: "models/".into(),
//!     num_threads: 4,
//!     ..Default::default()
//! };
//!
//! let engine = SnapOcrEngine::new(config).unwrap();
//!
//! // From raw RGBA pixels (e.g., from a screen capture)
//! let rgba_data: Vec<u8> = vec![128; 800 * 600 * 4];
//! let result = engine.recognize_rgba(&rgba_data, 800, 600).unwrap();
//! println!("Recognized text:\n{}", result.text);
//!
//! // From a PNG/JPEG file
//! let png_bytes = std::fs::read("screenshot.png").unwrap();
//! let result = engine.recognize_encoded_image(&png_bytes).unwrap();
//! for line in &result.lines {
//!     println!("[{:.2}] {}", line.score, line.text);
//! }
//! println!("Total latency: {:.1}ms", result.latency_ms.total_ms);
//! ```
//!
//! ## Architecture
//! ```text
//! ┌─────────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐     ┌─────────────┐
//! │ RGBA/PNG    │────>│ Preproc  │────>│ DBNet Det  │────>│ Affine   │────>│ CRNN Rec    │
//! │ Input       │     │ Resize   │     │ Contours   │     │ Crop     │     │ CTC Decode  │
//! └─────────────┘     │ Normalize│     │ MinAreaRect│     │ Lines    │     │ Batch       │
//!                     └──────────┘     └────────────┘     └──────────┘     └──────┬──────┘
//!                                                                                  │
//!                     ┌─────────────────────────────────────────────────────────────┘
//!                     │
//!                     v
//!              ┌──────────────┐     ┌─────────────┐
//!              │ Paragraph    │────>│ OcrResult   │
//!              │ Reorder +    │     │ {text,lines │
//!              │ Text Cleanup │     │  latency}   │
//!              └──────────────┘     └─────────────┘
//! ```

// Internal modules
mod config;
mod crop;
mod detect;
mod engine;
mod error;
mod postprocess;
mod preprocess;
mod recognize;
mod types;

// Optional C-ABI FFI exports (only compiled when building cdylib)
pub mod ffi;

// Public API re-exports
pub use config::OcrEngineConfig;
pub use engine::SnapOcrEngine;
pub use error::OcrError;
pub use types::{OcrBoundingBox, OcrLatencyBreakdown, OcrLineResult, OcrResult};

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static ORT_INITIALIZED: OnceLock<bool> = OnceLock::new();

/// Initialize the ONNX Runtime dynamic library from candidate paths.
pub fn init_ort_runtime(custom_dll_path: Option<&Path>) -> Result<(), OcrError> {
    ORT_INITIALIZED.get_or_init(|| {
        let candidates = [
            custom_dll_path.map(|p| p.to_path_buf()),
            std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("onnxruntime.dll"))),
            Some(PathBuf::from("onnxruntime.dll")),
            Some(PathBuf::from(r"D:\Python\Lib\site-packages\onnxruntime\capi\onnxruntime.dll")),
        ];

        for candidate in candidates.into_iter().flatten() {
            if candidate.exists() {
                if ort::init_from(&candidate).is_ok() {
                    log::info!("Successfully initialized ORT from {}", candidate.display());
                    return true;
                }
            }
        }

        // Try default init
        ort::init().commit()
    });

    Ok(())
}
