//! Public types for SnapOCR recognition results.
//!
//! These types follow the interface specification defined in PRD §5.1,
//! providing structured output for text detection and recognition results.

use serde::{Deserialize, Serialize};

/// A quadrilateral bounding box defined by 4 corner points.
///
/// Points are ordered: `[TopLeft, TopRight, BottomRight, BottomLeft]`.
/// Each point is `[x, y]` in pixel coordinates of the original input image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrBoundingBox {
    /// Four corner points: `[[x,y]; 4]` in order TL, TR, BR, BL
    pub points: [[f32; 2]; 4],
}

impl OcrBoundingBox {
    /// Create a new bounding box from 4 corner points.
    pub fn new(points: [[f32; 2]; 4]) -> Self {
        Self { points }
    }

    /// Create an axis-aligned bounding box from (x, y, width, height).
    pub fn from_rect(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            points: [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ],
        }
    }

    /// Calculate the center point of the bounding box.
    pub fn center(&self) -> [f32; 2] {
        let cx = self.points.iter().map(|p| p[0]).sum::<f32>() / 4.0;
        let cy = self.points.iter().map(|p| p[1]).sum::<f32>() / 4.0;
        [cx, cy]
    }

    /// Estimate the width of the bounding box (average of top and bottom edges).
    pub fn width(&self) -> f32 {
        let top = ((self.points[1][0] - self.points[0][0]).powi(2)
            + (self.points[1][1] - self.points[0][1]).powi(2))
        .sqrt();
        let bottom = ((self.points[2][0] - self.points[3][0]).powi(2)
            + (self.points[2][1] - self.points[3][1]).powi(2))
        .sqrt();
        (top + bottom) / 2.0
    }

    /// Estimate the height of the bounding box (average of left and right edges).
    pub fn height(&self) -> f32 {
        let left = ((self.points[3][0] - self.points[0][0]).powi(2)
            + (self.points[3][1] - self.points[0][1]).powi(2))
        .sqrt();
        let right = ((self.points[2][0] - self.points[1][0]).powi(2)
            + (self.points[2][1] - self.points[1][1]).powi(2))
        .sqrt();
        (left + right) / 2.0
    }

    /// Scale all points by the given factors.
    pub fn scale(&mut self, sx: f32, sy: f32) {
        for p in &mut self.points {
            p[0] *= sx;
            p[1] *= sy;
        }
    }
}

/// Recognition result for a single detected text line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrLineResult {
    /// Recognized text content for this line.
    pub text: String,
    /// Bounding box coordinates in the original image.
    pub box_coords: OcrBoundingBox,
    /// Recognition confidence score (0.0 ~ 1.0).
    pub score: f32,
}

/// Breakdown of inference latency for profiling and optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrLatencyBreakdown {
    /// Preprocessing time in milliseconds (grayscale, resize, normalize).
    pub preprocess_ms: f32,
    /// Text detection (DBNet) inference time in milliseconds.
    pub detect_ms: f32,
    /// Text line cropping and affine transform time in milliseconds.
    pub crop_ms: f32,
    /// Text recognition (CRNN+CTC) inference time in milliseconds.
    pub recognize_ms: f32,
    /// Post-processing time in milliseconds (paragraph reorder, text cleanup).
    pub postprocess_ms: f32,
    /// Total end-to-end time in milliseconds.
    pub total_ms: f32,
}

/// Complete OCR recognition result for a single image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// Full natural text after paragraph reordering and formatting.
    /// Paragraphs are separated by double newlines; lines within a paragraph
    /// are separated by single newlines.
    pub text: String,
    /// Per-line detailed results with bounding boxes and confidence scores.
    pub lines: Vec<OcrLineResult>,
    /// Inference latency breakdown for each pipeline stage.
    pub latency_ms: OcrLatencyBreakdown,
}

/// Internal representation of a detected text region before recognition.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct DetectedBox {
    /// Four corner points of the rotated bounding box.
    pub points: [[f32; 2]; 4],
    /// Detection confidence score (mean probability within the box region).
    pub score: f32,
}

#[allow(dead_code)]
impl DetectedBox {
    /// Convert to the public `OcrBoundingBox` type.
    pub fn to_bounding_box(&self) -> OcrBoundingBox {
        OcrBoundingBox::new(self.points)
    }

    /// Estimate the width (average of top and bottom edges).
    pub fn width(&self) -> f32 {
        let top = ((self.points[1][0] - self.points[0][0]).powi(2)
            + (self.points[1][1] - self.points[0][1]).powi(2))
        .sqrt();
        let bottom = ((self.points[2][0] - self.points[3][0]).powi(2)
            + (self.points[2][1] - self.points[3][1]).powi(2))
        .sqrt();
        (top + bottom) / 2.0
    }

    /// Estimate the height (average of left and right edges).
    pub fn height(&self) -> f32 {
        let left = ((self.points[3][0] - self.points[0][0]).powi(2)
            + (self.points[3][1] - self.points[0][1]).powi(2))
        .sqrt();
        let right = ((self.points[2][0] - self.points[1][0]).powi(2)
            + (self.points[2][1] - self.points[1][1]).powi(2))
        .sqrt();
        (left + right) / 2.0
    }
}
