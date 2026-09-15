//! CRNN (Convolutional Recurrent Neural Network) text recognition.
//!
//! Runs the recognition ONNX model on cropped text line images
//! and decodes the output using CTC greedy decoding.

use crate::config::OcrEngineConfig;
use crate::error::OcrError;
use crate::preprocess::preprocess_for_recognition;
use crate::recognize::ctc_decode::{ctc_greedy_decode, indices_to_string};

use image::RgbImage;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::sync::Mutex;

/// Result of recognizing a single text line.
#[derive(Debug, Clone)]
pub(crate) struct RecognizedLine {
    /// The decoded text string.
    pub text: String,
    /// Average CTC confidence score.
    pub confidence: f32,
}

/// CRNN text recognizer wrapping an ONNX Runtime session.
#[allow(dead_code)]
pub(crate) struct CrnnRecognizer {
    session: Mutex<Session>,
    vocab: Vec<char>,
    rec_image_height: u32,
    batch_size: usize,
}

impl CrnnRecognizer {
    /// Load the recognition model and vocabulary.
    pub fn load(config: &OcrEngineConfig) -> Result<Self, OcrError> {
        let model_path = config.model_dir.join("rec.onnx");
        let vocab_path = config.model_dir.join("vocab.txt");

        // Load vocabulary
        let vocab_text = std::fs::read_to_string(&vocab_path).map_err(|e| OcrError::VocabLoad {
            reason: format!("Failed to read {}: {}", vocab_path.display(), e),
        })?;

        // Build vocabulary: blank token at index 0, then one char per line
        let mut vocab: Vec<char> = vec!['\0']; // blank token
        for line in vocab_text.split('\n') {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if let Some(ch) = line.chars().next() {
                vocab.push(ch);
            }
        }
        // RapidOCR/PP-OCR appends a space token at the end if not present
        if vocab.len() == 6624 {
            vocab.push(' ');
        }
        log::info!("Loaded recognition vocabulary: {} tokens", vocab.len());

        // Note: For CRNN with dynamic sequence lengths, CPU inference is optimal
        // because DirectML recompiles HLSL shaders for every unique line width.
        let session = Session::builder()
            .map_err(|e| OcrError::ModelLoad { model_name: "rec.onnx".to_string(), reason: e.to_string() })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OcrError::ModelLoad { model_name: "rec.onnx".to_string(), reason: e.to_string() })?
            .with_intra_threads(config.num_threads)
            .map_err(|e| OcrError::ModelLoad { model_name: "rec.onnx".to_string(), reason: e.to_string() })?
            .commit_from_file(&model_path)
            .map_err(|e| OcrError::ModelLoad {
                model_name: "rec.onnx".to_string(),
                reason: e.to_string(),
            })?;

        Ok(Self {
            session: Mutex::new(session),
            vocab,
            rec_image_height: config.rec_image_height,
            batch_size: config.rec_batch_size,
        })
    }

    /// Recognize text from a list of cropped text line images.
    pub fn recognize_lines(
        &self,
        line_images: &[RgbImage],
    ) -> Result<Vec<RecognizedLine>, OcrError> {
        let mut results = Vec::with_capacity(line_images.len());

        for img in line_images {
            let result = self.recognize_single(img)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Recognize text from a single cropped text line image.
    fn recognize_single(&self, img: &RgbImage) -> Result<RecognizedLine, OcrError> {
        let input_tensor = preprocess_for_recognition(img, self.rec_image_height);
        let (_, c, h, w) = (
            input_tensor.shape()[0],
            input_tensor.shape()[1],
            input_tensor.shape()[2],
            input_tensor.shape()[3],
        );

        let tensor_data: Vec<f32> = input_tensor.iter().copied().collect();
        let rec_tensor = Tensor::from_array(([1, c, h, w], tensor_data))
            .map_err(|e| OcrError::InferenceError {
                stage: "recognition".to_string(),
                reason: format!("Failed to create tensor: {}", e),
            })?;

        let mut session = self.session.lock().map_err(|e| OcrError::InferenceError {
            stage: "recognition".to_string(),
            reason: format!("Session lock poisoned: {}", e),
        })?;

        let output = session
            .run(ort::inputs![rec_tensor])
            .map_err(|e| OcrError::InferenceError {
                stage: "recognition".to_string(),
                reason: e.to_string(),
            })?;

        let (rec_shape, logits_data) = output[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| OcrError::InferenceError {
                stage: "recognition_output".to_string(),
                reason: format!("Failed to extract output tensor: {}", e),
            })?;

        let (seq_len, vocab_size) = if rec_shape.len() == 3 {
            (rec_shape[1] as usize, rec_shape[2] as usize)
        } else if rec_shape.len() == 2 {
            (rec_shape[0] as usize, rec_shape[1] as usize)
        } else {
            return Err(OcrError::InferenceError {
                stage: "recognition_output".to_string(),
                reason: format!("Unexpected output shape: {:?}", rec_shape),
            });
        };

        // CTC greedy decode
        let decoded = ctc_greedy_decode(logits_data, seq_len, vocab_size, 0);

        // Map indices to characters
        let text = indices_to_string(&decoded, &self.vocab);

        Ok(RecognizedLine {
            text,
            confidence: decoded.confidence,
        })
    }

    /// Get the vocabulary size (including blank token).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_vocab_construction() {
        // Simulate a small vocabulary file
        let vocab_text = "a\nb\nc\n你\n好\n";
        let mut vocab: Vec<char> = vec![' ']; // blank
        for line in vocab_text.lines() {
            let line = line.trim();
            if !line.is_empty() {
                if let Some(ch) = line.chars().next() {
                    vocab.push(ch);
                }
            }
        }
        assert_eq!(vocab.len(), 6); // blank + 5 chars
        assert_eq!(vocab[0], ' ');
        assert_eq!(vocab[1], 'a');
        assert_eq!(vocab[5], '好');
    }
}
