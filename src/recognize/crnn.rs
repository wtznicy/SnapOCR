//! CRNN (Convolutional Recurrent Neural Network) text recognition.
//!
//! Runs the recognition ONNX model on cropped text line images
//! and decodes the output using CTC greedy decoding.
//! Supports multi-threaded parallel inference via Rayon and SessionPool.

use crate::config::OcrEngineConfig;
use crate::error::OcrError;
use crate::recognize::ctc_decode::{ctc_greedy_decode, indices_to_string};

use image::RgbImage;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use rayon::prelude::*;
use std::sync::{Condvar, Mutex};

/// A thread-safe pool of ONNX Runtime Sessions for parallel recognition.
struct SessionPool {
    sessions: Mutex<Vec<Session>>,
    cvar: Condvar,
}

impl SessionPool {
    fn new(sessions: Vec<Session>) -> Self {
        Self {
            sessions: Mutex::new(sessions),
            cvar: Condvar::new(),
        }
    }

    fn acquire(&self) -> PooledSession<'_> {
        let mut lock = self.sessions.lock().unwrap();
        while lock.is_empty() {
            lock = self.cvar.wait(lock).unwrap();
        }
        let session = lock.pop().unwrap();
        PooledSession {
            pool: self,
            session: Some(session),
        }
    }
}

struct PooledSession<'a> {
    pool: &'a SessionPool,
    session: Option<Session>,
}

impl<'a> std::ops::Deref for PooledSession<'a> {
    type Target = Session;
    fn deref(&self) -> &Self::Target {
        self.session.as_ref().unwrap()
    }
}

impl<'a> std::ops::DerefMut for PooledSession<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.as_mut().unwrap()
    }
}

impl<'a> Drop for PooledSession<'a> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let mut lock = self.pool.sessions.lock().unwrap();
            lock.push(session);
            self.pool.cvar.notify_one();
        }
    }
}

/// Result of recognizing a single text line.
#[derive(Debug, Clone)]
pub(crate) struct RecognizedLine {
    /// The decoded text string.
    pub text: String,
    /// Average CTC confidence score.
    pub confidence: f32,
}

/// CRNN text recognizer wrapping an ONNX Runtime session pool.
#[allow(dead_code)]
pub(crate) struct CrnnRecognizer {
    pool: SessionPool,
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

        // Configure SessionPool:
        // 4 concurrent worker sessions balances CPU throughput, L2/L3 cache locality, and RAM
        let pool_size = 4;

        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let session = Session::builder()
                .map_err(|e| OcrError::ModelLoad { model_name: format!("rec.onnx #{}", i), reason: e.to_string() })?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| OcrError::ModelLoad { model_name: format!("rec.onnx #{}", i), reason: e.to_string() })?
                .with_intra_threads(1)
                .map_err(|e| OcrError::ModelLoad { model_name: format!("rec.onnx #{}", i), reason: e.to_string() })?
                .commit_from_file(&model_path)
                .map_err(|e| OcrError::ModelLoad {
                    model_name: "rec.onnx".to_string(),
                    reason: e.to_string(),
                })?;
            sessions.push(session);
        }
        log::info!("Initialized CRNN recognizer session pool with {} instances", sessions.len());

        Ok(Self {
            pool: SessionPool::new(sessions),
            vocab,
            rec_image_height: config.rec_image_height,
            batch_size: config.rec_batch_size,
        })
    }

    /// Recognize text from a list of cropped text line images in parallel using Rayon.
    pub fn recognize_lines(
        &self,
        line_images: &[RgbImage],
    ) -> Result<Vec<RecognizedLine>, OcrError> {
        if line_images.is_empty() {
            return Ok(Vec::new());
        }

        line_images
            .par_iter()
            .map(|img| self.recognize_single(img))
            .collect()
    }

    /// Recognize text from a single cropped text line image.
    fn recognize_single(&self, img: &RgbImage) -> Result<RecognizedLine, OcrError> {
        let (w, h, data) = crate::preprocess::preprocess_for_recognition_vec(img, self.rec_image_height);
        let rec_tensor = Tensor::from_array(([1, 3, h as usize, w as usize], data))
            .map_err(|e| OcrError::InferenceError {
                stage: "recognition".to_string(),
                reason: format!("Failed to create tensor: {}", e),
            })?;

        let mut session = self.pool.acquire();

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
        let vocab_text = "a\nb\nc\n你\n好\n";
        let mut vocab: Vec<char> = vec![' '];
        for line in vocab_text.lines() {
            let line = line.trim();
            if !line.is_empty() {
                if let Some(ch) = line.chars().next() {
                    vocab.push(ch);
                }
            }
        }
        assert_eq!(vocab.len(), 6);
        assert_eq!(vocab[0], ' ');
        assert_eq!(vocab[1], 'a');
        assert_eq!(vocab[5], '好');
    }
}
