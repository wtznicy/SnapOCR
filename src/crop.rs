//! Perspective/affine transformation for cropping detected text line regions.
//!
//! Takes rotated bounding boxes from DBNet detection and warps each text line
//! into a horizontal, fixed-height image strip suitable for CRNN recognition.

use image::{Rgb, RgbImage};

/// Crop and perspective-warp a text line region from the source image.
///
/// Given a rotated quadrilateral defined by 4 corner points (TL, TR, BR, BL),
/// this function computes a perspective transformation to rectify the text
/// line into a horizontal image strip.
///
/// # Arguments
/// * `src` — Source RGB image.
/// * `points` — Four corner points `[TL, TR, BR, BL]` as `[[x, y]; 4]`.
/// * `target_height` — Desired output height (e.g., 32 or 48 pixels).
///
/// # Returns
/// An RGB image of the rectified text line with height = `target_height`.
pub(crate) fn crop_text_line(
    src: &RgbImage,
    points: &[[f32; 2]; 4],
    target_height: u32,
) -> RgbImage {
    // Calculate target dimensions from source quadrilateral
    let src_width = edge_length(&points[0], &points[1])
        .max(edge_length(&points[3], &points[2]))
        .round() as u32;
    let src_height = edge_length(&points[0], &points[3])
        .max(edge_length(&points[1], &points[2]))
        .round() as u32;

    let target_width = ((target_height as f32 * src_width as f32) / src_height.max(1) as f32).round() as u32;
    let target_width = target_width.max(1);

    // Destination rectangle corners
    let dst_points: [[f32; 2]; 4] = [
        [0.0, 0.0],                                    // TL
        [target_width as f32, 0.0],                     // TR
        [target_width as f32, target_height as f32],   // BR
        [0.0, target_height as f32],                    // BL
    ];

    // Compute the inverse perspective transformation matrix
    // (maps destination pixels back to source pixels)
    let inv_transform = compute_perspective_transform(&dst_points, points);

    // Apply the transformation with bilinear interpolation
    let mut output = RgbImage::new(target_width, target_height);

    let (sw, sh) = (src.width() as f32, src.height() as f32);

    for y in 0..target_height {
        for x in 0..target_width {
            let (sx, sy) = apply_perspective(&inv_transform, x as f32, y as f32);

            // Bilinear interpolation from source
            let rgb = bilinear_sample_rgb(src, sx, sy, sw, sh);
            output.put_pixel(x, y, Rgb(rgb));
        }
    }

    output
}

/// Batch crop multiple text line regions from the source image.
///
/// # Arguments
/// * `src` — Source RGB image.
/// * `boxes` — List of rotated bounding boxes, each as `[[x, y]; 4]`.
/// * `target_height` — Desired output height.
///
/// # Returns
/// A vector of (cropped image, original width) pairs, sorted by width (descending)
/// to facilitate efficient batching during recognition.
#[allow(dead_code)]
pub(crate) fn batch_crop_text_lines(
    src: &RgbImage,
    boxes: &[[[f32; 2]; 4]],
    target_height: u32,
) -> Vec<(RgbImage, usize)> {
    let mut crops: Vec<(RgbImage, usize)> = boxes
        .iter()
        .enumerate()
        .map(|(idx, pts)| {
            let cropped = crop_text_line(src, pts, target_height);
            (cropped, idx)
        })
        .collect();

    // Sort by width descending for optimal batching (padding efficiency)
    crops.sort_by(|a, b| b.0.width().cmp(&a.0.width()));
    crops
}

/// Compute the 3×3 perspective transformation matrix from 4 source points to 4 destination points.
///
/// Solves the system of 8 linear equations derived from the projective mapping:
///
/// ```text
/// dx = (a*sx + b*sy + c) / (g*sx + h*sy + 1)
/// dy = (d*sx + e*sy + f) / (g*sx + h*sy + 1)
/// ```
///
/// Returns a 3×3 matrix `[a, b, c, d, e, f, g, h, 1]` stored row-major as `[f32; 9]`.
fn compute_perspective_transform(
    src: &[[f32; 2]; 4],
    dst: &[[f32; 2]; 4],
) -> [f32; 9] {
    let mut a_mat = [[0.0f64; 8]; 8];
    let mut b_vec = [0.0f64; 8];

    for i in 0..4 {
        let (sx, sy) = (src[i][0] as f64, src[i][1] as f64);
        let (dx, dy) = (dst[i][0] as f64, dst[i][1] as f64);

        let row0 = i * 2;
        let row1 = i * 2 + 1;

        a_mat[row0] = [sx, sy, 1.0, 0.0, 0.0, 0.0, -sx * dx, -sy * dx];
        b_vec[row0] = dx;

        a_mat[row1] = [0.0, 0.0, 0.0, sx, sy, 1.0, -sx * dy, -sy * dy];
        b_vec[row1] = dy;
    }

    // Solve using Gaussian elimination with partial pivoting
    let coeffs = solve_8x8(&mut a_mat, &mut b_vec);

    [
        coeffs[0] as f32,
        coeffs[1] as f32,
        coeffs[2] as f32,
        coeffs[3] as f32,
        coeffs[4] as f32,
        coeffs[5] as f32,
        coeffs[6] as f32,
        coeffs[7] as f32,
        1.0,
    ]
}

/// Solve an 8×8 linear system using Gaussian elimination with partial pivoting.
fn solve_8x8(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> [f64; 8] {
    let n = 8;

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = a[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];
        if pivot.abs() < 1e-12 {
            // Singular matrix — return identity-like result
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = [0.0f64; 8];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    x
}

/// Apply a 3×3 perspective transformation to a point.
#[inline]
fn apply_perspective(m: &[f32; 9], x: f32, y: f32) -> (f32, f32) {
    let w = m[6] * x + m[7] * y + m[8];
    let w = if w.abs() < 1e-10 { 1.0 } else { w };
    let sx = (m[0] * x + m[1] * y + m[2]) / w;
    let sy = (m[3] * x + m[4] * y + m[5]) / w;
    (sx, sy)
}

/// Sample an RGB pixel from an RGB image using bilinear interpolation.
#[inline]
fn bilinear_sample_rgb(src: &RgbImage, x: f32, y: f32, sw: f32, sh: f32) -> [u8; 3] {
    let x = x.clamp(0.0, sw - 1.001);
    let y = y.clamp(0.0, sh - 1.001);

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(sw as u32 - 1);
    let y1 = (y0 + 1).min(sh as u32 - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = src.get_pixel(x0, y0).0;
    let p10 = src.get_pixel(x1, y0).0;
    let p01 = src.get_pixel(x0, y1).0;
    let p11 = src.get_pixel(x1, y1).0;

    let mut out = [0u8; 3];
    for c in 0..3 {
        let val = p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
            + p10[c] as f32 * fx * (1.0 - fy)
            + p01[c] as f32 * (1.0 - fx) * fy
            + p11[c] as f32 * fx * fy;
        out[c] = val.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Compute the Euclidean distance between two points.
#[inline]
fn edge_length(a: &[f32; 2], b: &[f32; 2]) -> f32 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_length() {
        assert!((edge_length(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_perspective_identity() {
        // Same source and destination points → identity-like transform
        let pts: [[f32; 2]; 4] = [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 50.0],
            [0.0, 50.0],
        ];
        let m = compute_perspective_transform(&pts, &pts);

        let (rx, ry) = apply_perspective(&m, 50.0, 25.0);
        assert!((rx - 50.0).abs() < 0.1, "rx = {}", rx);
        assert!((ry - 25.0).abs() < 0.1, "ry = {}", ry);
    }

    #[test]
    fn test_crop_text_line_basic() {
        // Create a simple 100x100 test image
        let src = RgbImage::from_fn(100, 100, |x, y| {
            if x >= 10 && x < 90 && y >= 20 && y < 40 {
                image::Rgb([255, 255, 255]) // White text region
            } else {
                image::Rgb([0, 0, 0]) // Black background
            }
        });

        let points: [[f32; 2]; 4] = [
            [10.0, 20.0],
            [90.0, 20.0],
            [90.0, 40.0],
            [10.0, 40.0],
        ];

        let cropped = crop_text_line(&src, &points, 32);
        assert_eq!(cropped.height(), 32);
        assert!(cropped.width() > 0);
    }
}
