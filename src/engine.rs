//! Core OCR engine that orchestrates the complete pipeline.
//!
//! Pipeline flow:
//! ```text
//! RGBA/PNG input
//!   → Preprocessing (resize, normalize, NCHW tensor)
//!   → Detection (DBNet inference + contour extraction)
//!   → Cropping (perspective warp each text line)
//!   → Recognition (CRNN inference + CTC decode per line)
//!   → Post-processing (paragraph reorder + text cleanup)
//!   → OcrResult output
//! ```

use std::time::Instant;

use image::DynamicImage;

use crate::config::OcrEngineConfig;
use crate::crop::crop_text_line;
use crate::detect::dbnet::DbNetDetector;
use crate::error::OcrError;
use crate::postprocess::paragraph::reorder_into_paragraphs;
use crate::postprocess::text_clean::clean_text;
use crate::preprocess::{decode_encoded, decode_rgba, preprocess_for_detection};
use crate::recognize::crnn::CrnnRecognizer;
use crate::types::{OcrLatencyBreakdown, OcrLineResult, OcrResult};

/// The main SnapOCR engine.
///
/// Holds loaded models and configuration. Thread-safe for concurrent use
/// (models are read-only after loading).
///
/// # Example
/// ```no_run
/// use snap_ocr::{SnapOcrEngine, OcrEngineConfig};
///
/// let config = OcrEngineConfig {
///     model_dir: "models/".into(),
///     ..Default::default()
/// };
/// let engine = SnapOcrEngine::new(config).unwrap();
///
/// // Recognize from RGBA pixels
/// let rgba_data: Vec<u8> = vec![0; 1920 * 1080 * 4];
/// let result = engine.recognize_rgba(&rgba_data, 1920, 1080).unwrap();
/// println!("{}", result.text);
/// ```
pub struct SnapOcrEngine {
    detector: DbNetDetector,
    recognizer: CrnnRecognizer,
    max_side_len: u32,
    rec_image_height: u32,
}

impl SnapOcrEngine {
    /// Initialize the engine and preload models.
    ///
    /// This loads both the detection and recognition ONNX models,
    /// the vocabulary file, and optimizes them for inference.
    ///
    /// # Arguments
    /// * `config` — Engine configuration (model paths, thresholds, etc.)
    ///
    /// # Errors
    /// Returns `OcrError::ModelLoad` if any model file cannot be loaded.
    pub fn new(config: OcrEngineConfig) -> Result<Self, OcrError> {
        crate::init_ort_runtime(None)?;
        log::info!("Initializing SnapOCR engine from {:?}", config.model_dir);

        let detector = DbNetDetector::load(&config)?;
        log::info!("Detection model loaded");

        let recognizer = CrnnRecognizer::load(&config)?;
        log::info!(
            "Recognition model loaded (vocab size: {})",
            recognizer.vocab_size()
        );

        Ok(Self {
            detector,
            max_side_len: config.max_side_len,
            rec_image_height: config.rec_image_height,
            recognizer,
        })
    }

    /// Recognize text from raw RGBA pixel data (zero-copy interface).
    ///
    /// # Arguments
    /// * `rgba_bytes` — Raw RGBA pixel data, length must be `width * height * 4`.
    /// * `width` — Image width in pixels.
    /// * `height` — Image height in pixels.
    ///
    /// # Returns
    /// `OcrResult` containing the recognized text, per-line details, and latency breakdown.
    pub fn recognize_rgba(
        &self,
        rgba_bytes: &[u8],
        width: u32,
        height: u32,
    ) -> Result<OcrResult, OcrError> {
        let total_start = Instant::now();

        // === Stage 1: Decode RGBA ===
        let img = decode_rgba(rgba_bytes, width, height)?;

        self.run_pipeline(img, total_start)
    }

    /// Recognize text from a PNG/JPEG compressed image byte stream.
    ///
    /// # Arguments
    /// * `img_bytes` — PNG or JPEG compressed image data.
    ///
    /// # Returns
    /// `OcrResult` containing the recognized text, per-line details, and latency breakdown.
    pub fn recognize_encoded_image(&self, img_bytes: &[u8]) -> Result<OcrResult, OcrError> {
        let total_start = Instant::now();

        // === Stage 1: Decode image ===
        let img = decode_encoded(img_bytes)?;

        self.run_pipeline(img, total_start)
    }

    /// Run the full OCR pipeline on a decoded image.
    fn run_pipeline(
        &self,
        img: DynamicImage,
        total_start: Instant,
    ) -> Result<OcrResult, OcrError> {
        // === Stage 2: Preprocessing ===
        let preprocess_start = Instant::now();
        let preprocessed = preprocess_for_detection(&img, self.max_side_len)?;
        let preprocess_ms = preprocess_start.elapsed().as_secs_f32() * 1000.0;

        // === Stage 3: Text Detection ===
        let detect_start = Instant::now();
        let detected_boxes = self.detector.detect(&preprocessed)?;
        let detect_ms = detect_start.elapsed().as_secs_f32() * 1000.0;

        log::debug!("Detected {} text regions", detected_boxes.len());

        // Handle empty detection
        if detected_boxes.is_empty() {
            let total_ms = total_start.elapsed().as_secs_f32() * 1000.0;
            return Ok(OcrResult {
                text: String::new(),
                lines: Vec::new(),
                latency_ms: OcrLatencyBreakdown {
                    preprocess_ms,
                    detect_ms,
                    crop_ms: 0.0,
                    recognize_ms: 0.0,
                    postprocess_ms: 0.0,
                    total_ms,
                },
            });
        }

        // === Stage 4: Crop text lines ===
        let crop_start = Instant::now();
        let src_rgb = img.to_rgb8();

        let line_images: Vec<image::RgbImage> = detected_boxes
            .iter()
            .map(|db| crop_text_line(&src_rgb, &db.points, self.rec_image_height))
            .collect();
        let crop_ms = crop_start.elapsed().as_secs_f32() * 1000.0;

        // === Stage 5: Text Recognition ===
        let recognize_start = Instant::now();
        let recognized = self.recognizer.recognize_lines(&line_images)?;
        let recognize_ms = recognize_start.elapsed().as_secs_f32() * 1000.0;

        // === Stage 6: Post-processing ===
        let postprocess_start = Instant::now();

        // Build line results
        let mut line_results: Vec<OcrLineResult> = detected_boxes
            .iter()
            .zip(recognized.iter())
            .filter(|(_, rec)| !rec.text.is_empty())
            .map(|(det, rec)| {
                let cleaned_text = clean_text(&rec.text);
                OcrLineResult {
                    text: cleaned_text,
                    box_coords: det.to_bounding_box(),
                    score: rec.confidence,
                }
            })
            .collect();

        // Paragraph geometric reordering
        let full_text = reorder_into_paragraphs(&mut line_results);

        let postprocess_ms = postprocess_start.elapsed().as_secs_f32() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f32() * 1000.0;

        log::info!(
            "OCR complete: {} lines, {:.1}ms (pre:{:.1} det:{:.1} crop:{:.1} rec:{:.1} post:{:.1})",
            line_results.len(),
            total_ms,
            preprocess_ms,
            detect_ms,
            crop_ms,
            recognize_ms,
            postprocess_ms,
        );

        Ok(OcrResult {
            text: full_text,
            lines: line_results,
            latency_ms: OcrLatencyBreakdown {
                preprocess_ms,
                detect_ms,
                crop_ms,
                recognize_ms,
                postprocess_ms,
                total_ms,
            },
        })
    }
}
