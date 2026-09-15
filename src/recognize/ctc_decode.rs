//! CTC (Connectionist Temporal Classification) greedy decoding.
//!
//! Decodes the raw logits output from the CRNN recognition model
//! into character sequences with confidence scores.

/// Result of decoding a single text line from CTC logits.
#[derive(Debug, Clone)]
pub(crate) struct CtcDecodedLine {
    /// Decoded character indices (mapped to vocabulary).
    pub char_indices: Vec<usize>,
    /// Average confidence score (mean of per-step max softmax probabilities).
    pub confidence: f32,
}

/// Perform greedy CTC decoding on a logits matrix for a single sequence.
///
/// # Algorithm
/// 1. For each time step, find the character index with the highest logit value.
/// 2. Collapse consecutive duplicate indices.
/// 3. Remove blank tokens (index 0 by convention).
/// 4. Compute average confidence from the softmax probabilities of selected chars.
///
/// # Arguments
/// * `logits` — Raw model output, shape `[seq_len, vocab_size]`, stored as
///   a flat slice in row-major order.
/// * `seq_len` — Number of time steps (sequence length dimension).
/// * `vocab_size` — Size of the character vocabulary (including blank at index 0).
/// * `blank_index` — Index of the CTC blank token (typically 0).
pub(crate) fn ctc_greedy_decode(
    logits: &[f32],
    seq_len: usize,
    vocab_size: usize,
    blank_index: usize,
) -> CtcDecodedLine {
    assert_eq!(
        logits.len(),
        seq_len * vocab_size,
        "Logits size mismatch: expected {}x{} = {}, got {}",
        seq_len,
        vocab_size,
        seq_len * vocab_size,
        logits.len()
    );

    let mut char_indices = Vec::with_capacity(seq_len);
    let mut confidences = Vec::with_capacity(seq_len);
    let mut prev_index = blank_index;

    for t in 0..seq_len {
        let row_start = t * vocab_size;
        let row = &logits[row_start..row_start + vocab_size];

        // Find argmax and compute softmax probability for this step
        let (max_idx, max_logit) = argmax(row);
        let softmax_prob = softmax_single(row, max_idx, max_logit);

        // CTC collapsing: skip if same as previous, or if blank
        if max_idx != blank_index && max_idx != prev_index {
            char_indices.push(max_idx);
            confidences.push(softmax_prob);
        }

        prev_index = max_idx;
    }

    let confidence = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    CtcDecodedLine {
        char_indices,
        confidence,
    }
}

/// Perform greedy CTC decoding for a batch of sequences.
///
/// # Arguments
/// * `batch_logits` — Flat logits for the entire batch,
///   shape `[batch_size, seq_len, vocab_size]` in row-major order.
/// * `batch_size` — Number of sequences in the batch.
/// * `seq_len` — Sequence length (same for all items in batch, after padding).
/// * `vocab_size` — Vocabulary size including blank.
/// * `blank_index` — CTC blank token index.
#[allow(dead_code)]
pub(crate) fn ctc_greedy_decode_batch(
    batch_logits: &[f32],
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    blank_index: usize,
) -> Vec<CtcDecodedLine> {
    let step = seq_len * vocab_size;
    (0..batch_size)
        .map(|b| {
            let offset = b * step;
            let logits = &batch_logits[offset..offset + step];
            ctc_greedy_decode(logits, seq_len, vocab_size, blank_index)
        })
        .collect()
}

/// Map decoded character indices to actual characters using a vocabulary.
///
/// # Arguments
/// * `decoded` — The CTC decoded result containing character indices.
/// * `vocab` — The vocabulary (list of characters), where `vocab[i]` is the
///   character for model output index `i`. Note: index 0 is typically reserved
///   for the CTC blank token.
pub(crate) fn indices_to_string(decoded: &CtcDecodedLine, vocab: &[char]) -> String {
    decoded
        .char_indices
        .iter()
        .filter_map(|&idx| {
            if idx < vocab.len() {
                Some(vocab[idx])
            } else {
                None // Skip out-of-vocabulary indices
            }
        })
        .collect()
}

/// Find the index and value of the maximum element in a slice.
#[inline]
fn argmax(row: &[f32]) -> (usize, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    (max_idx, max_val)
}

/// Compute the softmax probability for a single element.
///
/// Uses the log-sum-exp trick for numerical stability:
/// `softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))`
#[inline]
fn softmax_single(row: &[f32], _target_idx: usize, max_logit: f32) -> f32 {
    let sum_exp: f32 = row.iter().map(|&v| (v - max_logit).exp()).sum();
    1.0 / sum_exp // exp(max - max) / sum = 1 / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_decode_simple() {
        // 3 time steps, vocab_size = 4 (blank=0, a=1, b=2, c=3)
        // Step 0: a (index 1), Step 1: a (index 1), Step 2: b (index 2)
        // After CTC collapsing: "ab" (indices [1, 2])
        let logits = vec![
            // step 0: blank=-10, a=10, b=-10, c=-10
            -10.0, 10.0, -10.0, -10.0,
            // step 1: blank=-10, a=10, b=-10, c=-10  (duplicate, should collapse)
            -10.0, 10.0, -10.0, -10.0,
            // step 2: blank=-10, a=-10, b=10, c=-10
            -10.0, -10.0, 10.0, -10.0,
        ];

        let result = ctc_greedy_decode(&logits, 3, 4, 0);
        assert_eq!(result.char_indices, vec![1, 2]);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_greedy_decode_with_blanks() {
        // a, blank, a → "aa" (blank separates the two a's)
        let logits = vec![
            -10.0, 10.0, -10.0, -10.0, // a
            10.0, -10.0, -10.0, -10.0,  // blank
            -10.0, 10.0, -10.0, -10.0,  // a
        ];

        let result = ctc_greedy_decode(&logits, 3, 4, 0);
        assert_eq!(result.char_indices, vec![1, 1]);
    }

    #[test]
    fn test_greedy_decode_all_blanks() {
        let logits = vec![
            10.0, -10.0, -10.0, -10.0,
            10.0, -10.0, -10.0, -10.0,
            10.0, -10.0, -10.0, -10.0,
        ];

        let result = ctc_greedy_decode(&logits, 3, 4, 0);
        assert!(result.char_indices.is_empty());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_indices_to_string() {
        let decoded = CtcDecodedLine {
            char_indices: vec![1, 2, 3],
            confidence: 0.95,
        };
        let vocab: Vec<char> = vec![' ', 'a', 'b', 'c', 'd']; // blank at 0
        let text = indices_to_string(&decoded, &vocab);
        assert_eq!(text, "abc");
    }

    #[test]
    fn test_batch_decode() {
        let vocab_size = 4;
        let seq_len = 3;
        let batch_logits = vec![
            // Batch 0: "a"
            -10.0, 10.0, -10.0, -10.0,
            10.0, -10.0, -10.0, -10.0,
            10.0, -10.0, -10.0, -10.0,
            // Batch 1: "b"
            -10.0, -10.0, 10.0, -10.0,
            10.0, -10.0, -10.0, -10.0,
            10.0, -10.0, -10.0, -10.0,
        ];

        let results = ctc_greedy_decode_batch(&batch_logits, 2, seq_len, vocab_size, 0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].char_indices, vec![1]);
        assert_eq!(results[1].char_indices, vec![2]);
    }
}
