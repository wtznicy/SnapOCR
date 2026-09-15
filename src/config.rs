//! Configuration for the SnapOCR engine.

use std::path::PathBuf;

/// Configuration for initializing a `SnapOcrEngine` instance.
///
/// # Example
/// ```no_run
/// use snap_ocr::OcrEngineConfig;
///
/// let config = OcrEngineConfig {
///     model_dir: "models/".into(),
///     num_threads: 4,
///     ..Default::default()
/// };
/// ```
pub struct OcrEngineConfig {
    /// Path to the directory containing model files:
    /// - `det.onnx` — text detection model (DBNet)
    /// - `rec.onnx` — text recognition model (CRNN)
    /// - `vocab.txt` — character vocabulary (one character per line)
    pub model_dir: PathBuf,

    /// Number of inference threads.
    ///
    /// Recommended: set to the number of performance cores (2~4).
    /// On Intel 12th gen+ hybrid architectures, avoid using efficiency cores
    /// to prevent thread synchronization overhead.
    pub num_threads: usize,

    /// Maximum side length for detection input.
    ///
    /// The input image is proportionally scaled so that the longest side
    /// does not exceed this value. Dimensions are then padded to multiples
    /// of 32 (required by DBNet's stride).
    ///
    /// Lower values = faster detection but may miss small text.
    /// Recommended: 736 for clear screenshots, 960 for general use.
    pub max_side_len: u32,

    /// Minimum detection confidence threshold (0.0 ~ 1.0).
    ///
    /// Text regions with average probability below this threshold
    /// are discarded. Higher values reduce false positives but may
    /// miss faint text.
    pub score_thresh: f32,

    /// Detection box expansion ratio.
    ///
    /// After finding text region contours, the bounding box is expanded
    /// by this ratio to include surrounding context (e.g., descenders,
    /// diacritics). Typical value: 1.5 ~ 2.0.
    pub box_expand_ratio: f32,

    /// Minimum text region area in pixels (at detection scale).
    ///
    /// Regions smaller than this are discarded as noise.
    pub min_area: f32,

    /// Recognition model input height in pixels.
    ///
    /// Text line crops are resized to this height before recognition.
    /// Must match the recognition model's expected input height.
    pub rec_image_height: u32,

    /// Maximum batch size for recognition.
    ///
    /// Multiple text lines are batched together for more efficient
    /// inference. Larger batches use more memory but fewer inference calls.
    pub rec_batch_size: usize,

    /// Whether to enable hardware acceleration.
    ///
    /// When using `tract-onnx` (pure Rust), this option is ignored.
    /// When using `ort`, this enables DirectML (Windows), CoreML (macOS),
    /// or OpenVINO (Linux).
    pub enable_hw_accel: bool,
}

impl Default for OcrEngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models"),
            num_threads: 4,
            max_side_len: 960,
            score_thresh: 0.3,
            box_expand_ratio: 2.1,
            min_area: 3.0,
            rec_image_height: 48,
            rec_batch_size: 8,
            enable_hw_accel: false,
        }
    }
}
