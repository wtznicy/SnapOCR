//! DBNet (Differentiable Binarization) text detection.
//!
//! Runs the detection ONNX model and post-processes the probability map
//! to extract rotated text line bounding boxes.

use crate::config::OcrEngineConfig;
use crate::detect::contour::{
    expand_rotated_rect, find_contours, min_area_rect, simplify_contour,
};
use crate::error::OcrError;
use crate::preprocess::PreprocessedDetInput;
use crate::types::DetectedBox;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::sync::Mutex;

/// DBNet text detector wrapping an ONNX Runtime session.
pub(crate) struct DbNetDetector {
    session: Mutex<Session>,
    score_thresh: f32,
    box_expand_ratio: f32,
    min_area: f32,
}

impl DbNetDetector {
    /// Load the detection model from an ONNX file.
    pub fn load(config: &OcrEngineConfig) -> Result<Self, OcrError> {
        let model_path = config.model_dir.join("det.onnx");

        let mut builder = Session::builder()
            .map_err(|e| OcrError::ModelLoad { model_name: "det.onnx".to_string(), reason: e.to_string() })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OcrError::ModelLoad { model_name: "det.onnx".to_string(), reason: e.to_string() })?
            .with_intra_threads(config.num_threads)
            .map_err(|e| OcrError::ModelLoad { model_name: "det.onnx".to_string(), reason: e.to_string() })?;

        let session = if config.enable_hw_accel {
            match builder.with_execution_providers([ort::ep::DirectML::default().build()]) {
                Ok(mut b) => b.commit_from_file(&model_path),
                Err(_) => {
                    log::warn!("DirectML not available, falling back to CPU");
                    Session::builder()
                        .map_err(|e| OcrError::ModelLoad { model_name: "det.onnx".to_string(), reason: e.to_string() })?
                        .with_optimization_level(GraphOptimizationLevel::Level3)
                        .map_err(|e| OcrError::ModelLoad { model_name: "det.onnx".to_string(), reason: e.to_string() })?
                        .with_intra_threads(config.num_threads)
                        .map_err(|e| OcrError::ModelLoad { model_name: "det.onnx".to_string(), reason: e.to_string() })?
                        .commit_from_file(&model_path)
                }
            }
        } else {
            builder.commit_from_file(&model_path)
        }
        .map_err(|e| OcrError::ModelLoad {
            model_name: "det.onnx".to_string(),
            reason: e.to_string(),
        })?;

        Ok(Self {
            session: Mutex::new(session),
            score_thresh: config.score_thresh,
            box_expand_ratio: config.box_expand_ratio,
            min_area: config.min_area,
        })
    }

    /// Run text detection on a preprocessed image.
    ///
    /// Returns a list of detected text regions with rotated bounding boxes.
    pub fn detect(&self, input: &PreprocessedDetInput) -> Result<Vec<DetectedBox>, OcrError> {
        let (_, _, h, w) = (
            input.tensor.shape()[0],
            input.tensor.shape()[1],
            input.tensor.shape()[2],
            input.tensor.shape()[3],
        );

        let tensor_data: Vec<f32> = input.tensor.iter().copied().collect();
        let det_tensor = Tensor::from_array(([1, 3, h, w], tensor_data))
            .map_err(|e| OcrError::InferenceError {
                stage: "detection".to_string(),
                reason: format!("Failed to create tensor: {}", e),
            })?;

        let mut session = self.session.lock().map_err(|e| OcrError::InferenceError {
            stage: "detection".to_string(),
            reason: format!("Session lock poisoned: {}", e),
        })?;

        let result = session
            .run(ort::inputs![det_tensor])
            .map_err(|e| OcrError::InferenceError {
                stage: "detection".to_string(),
                reason: e.to_string(),
            })?;

        let (prob_shape, prob_data) = result[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| OcrError::InferenceError {
                stage: "detection_output".to_string(),
                reason: format!("Failed to extract output tensor: {}", e),
            })?;

        let map_h = prob_shape[2] as usize;
        let map_w = prob_shape[3] as usize;

        // Post-process: threshold → contours → rotated rects → filter
        let boxes = self.postprocess_probability_map(
            prob_data,
            map_w,
            map_h,
            input.scale_x,
            input.scale_y,
        );

        Ok(boxes)
    }

    /// Post-process the probability map to extract text bounding boxes.
    ///
    /// Steps:
    /// 1. Threshold the probability map to create a binary bitmap
    /// 2. Extract contours from the bitmap
    /// 3. For each contour, compute the minimum-area rotated rectangle
    /// 4. Filter by score (mean probability in the region) and area
    /// 5. Scale coordinates back to original image space
    fn postprocess_probability_map(
        &self,
        prob_map: &[f32],
        width: usize,
        height: usize,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<DetectedBox> {
        // Step 1: Binary thresholding
        let bitmap: Vec<u8> = prob_map
            .iter()
            .map(|&p| if p > self.score_thresh { 255 } else { 0 })
            .collect();

        // Step 2: Find contours
        let contours = find_contours(&bitmap, width, height);

        let mut boxes = Vec::new();

        for contour in &contours {
            if contour.len() < 4 {
                continue;
            }

            // Simplify the contour
            let epsilon = 2.0;
            let simplified = simplify_contour(contour, epsilon);
            if simplified.len() < 3 {
                continue;
            }

            // Step 3: Compute minimum-area rotated rectangle
            let rect = min_area_rect(&simplified);

            // Filter by minimum area
            if rect.area() < self.min_area {
                continue;
            }

            // Step 4: Compute score (mean probability within the rotated box)
            let rect_corners = rect.corners();
            let score = compute_box_score(prob_map, width, height, &rect_corners);
            if score < self.score_thresh {
                continue;
            }

            // Expand the rectangle slightly to include context
            let expanded = expand_rotated_rect(&rect, self.box_expand_ratio);

            // Step 5: Get corner points and scale back to original image coordinates
            let mut corners = expanded.corners();
            for corner in &mut corners {
                corner[0] /= scale_x;
                corner[1] /= scale_y;
            }

            // Order points: TL, TR, BR, BL
            let corners = order_points(corners);

            boxes.push(DetectedBox {
                points: corners,
                score,
            });
        }

        // Sort boxes top-to-bottom, left-to-right for consistent output
        boxes.sort_by(|a, b| {
            let ay = (a.points[0][1] + a.points[2][1]) / 2.0;
            let by = (b.points[0][1] + b.points[2][1]) / 2.0;
            ay.partial_cmp(&by).unwrap_or(std::cmp::Ordering::Equal)
        });

        boxes
    }
}

/// Compute the mean probability within a rotated box using scanline sampling.
fn compute_box_score(
    prob_map: &[f32],
    width: usize,
    height: usize,
    corners: &[[f32; 2]; 4],
) -> f32 {
    let min_x = corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min).max(0.0) as usize;
    let max_x = corners.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max).min(width as f32 - 1.0) as usize;
    let min_y = corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).max(0.0) as usize;
    let max_y = corners.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).min(height as f32 - 1.0) as usize;

    if min_x >= max_x || min_y >= max_y {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut count = 0u32;

    for y in min_y..=max_y {
        let row_offset = y * width;
        let y_f = y as f32;
        for x in min_x..=max_x {
            if point_in_quad(x as f32, y_f, corners) {
                sum += prob_map[row_offset + x];
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// Test if a point is inside a quadrilateral using ray casting.
#[inline]
fn point_in_quad(x: f32, y: f32, quad: &[[f32; 2]; 4]) -> bool {
    let mut inside = false;
    let mut j = 3;

    for i in 0..4 {
        let (xi, yi) = (quad[i][0], quad[i][1]);
        let (xj, yj) = (quad[j][0], quad[j][1]);

        if ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Order 4 corner points as [TopLeft, TopRight, BottomRight, BottomLeft].
pub(crate) fn order_points(points: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let mut sorted = points;
    sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    let (index_1, index_4) = if sorted[1][1] > sorted[0][1] {
        (0, 1)
    } else {
        (1, 0)
    };

    let (index_2, index_3) = if sorted[3][1] > sorted[2][1] {
        (2, 3)
    } else {
        (3, 2)
    };

    [
        sorted[index_1],
        sorted[index_2],
        sorted[index_3],
        sorted[index_4],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_points_axis_aligned() {
        let points: [[f32; 2]; 4] = [
            [100.0, 100.0], // BR
            [0.0, 0.0],     // TL
            [100.0, 0.0],   // TR
            [0.0, 100.0],   // BL
        ];
        let ordered = order_points(points);
        // TL
        assert_eq!(ordered[0], [0.0, 0.0]);
        // TR
        assert_eq!(ordered[1], [100.0, 0.0]);
        // BR
        assert_eq!(ordered[2], [100.0, 100.0]);
        // BL
        assert_eq!(ordered[3], [0.0, 100.0]);
    }

    #[test]
    fn test_point_in_quad() {
        let quad = [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];
        assert!(point_in_quad(5.0, 5.0, &quad));
        assert!(!point_in_quad(15.0, 5.0, &quad));
    }
}
