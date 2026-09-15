//! Error types for the SnapOCR engine.

use std::fmt;

/// All possible errors returned by the SnapOCR engine.
#[derive(Debug)]
pub enum OcrError {
    /// Failed to load an ONNX model file.
    ModelLoad {
        model_name: String,
        reason: String,
    },
    /// Invalid input data (e.g., wrong dimensions, empty buffer).
    InvalidInput {
        reason: String,
    },
    /// Error during ONNX model inference.
    InferenceError {
        stage: String,
        reason: String,
    },
    /// Failed to decode an image (PNG/JPEG).
    ImageDecode {
        reason: String,
    },
    /// Error during post-processing (paragraph reorder, text cleanup).
    PostProcess {
        reason: String,
    },
    /// Vocabulary/dictionary file error.
    VocabLoad {
        reason: String,
    },
    /// Generic I/O error.
    Io(std::io::Error),
}

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OcrError::ModelLoad { model_name, reason } => {
                write!(f, "Failed to load model '{}': {}", model_name, reason)
            }
            OcrError::InvalidInput { reason } => {
                write!(f, "Invalid input: {}", reason)
            }
            OcrError::InferenceError { stage, reason } => {
                write!(f, "Inference error in '{}': {}", stage, reason)
            }
            OcrError::ImageDecode { reason } => {
                write!(f, "Image decode error: {}", reason)
            }
            OcrError::PostProcess { reason } => {
                write!(f, "Post-processing error: {}", reason)
            }
            OcrError::VocabLoad { reason } => {
                write!(f, "Vocabulary load error: {}", reason)
            }
            OcrError::Io(e) => {
                write!(f, "I/O error: {}", e)
            }
        }
    }
}

impl std::error::Error for OcrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OcrError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OcrError {
    fn from(e: std::io::Error) -> Self {
        OcrError::Io(e)
    }
}

impl From<image::ImageError> for OcrError {
    fn from(e: image::ImageError) -> Self {
        OcrError::ImageDecode {
            reason: e.to_string(),
        }
    }
}
