//! Image preprocessing for the SnapOCR pipeline.
//!
//! Converts raw RGBA pixel buffers into normalized NCHW tensors
//! suitable for DBNet and CRNN model inference.

use crate::error::OcrError;
use image::{DynamicImage, RgbImage, imageops::FilterType};
use ndarray::Array4;

/// DBNet normalization constants (PaddleOCR / RapidOCR standard: (x/255 - 0.5) / 0.5).
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Result of preprocessing an image for detection.
#[allow(dead_code)]
pub(crate) struct PreprocessedDetInput {
    /// Normalized NCHW tensor `[1, 3, H, W]` in `f32`.
    pub tensor: Array4<f32>,
    /// Scale factor applied to the width dimension.
    pub scale_x: f32,
    /// Scale factor applied to the height dimension.
    pub scale_y: f32,
    /// Original image width before scaling.
    pub orig_width: u32,
    /// Original image height before scaling.
    pub orig_height: u32,
    /// The resized RGB image (kept for use in text line cropping).
    pub resized_rgb: RgbImage,
}

/// Decode an RGBA byte buffer into a `DynamicImage`.
///
/// # Arguments
/// * `rgba_bytes` — Raw RGBA pixel data, length must be `width * height * 4`.
/// * `width` — Image width in pixels.
/// * `height` — Image height in pixels.
pub(crate) fn decode_rgba(
    rgba_bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<DynamicImage, OcrError> {
    let expected_len = (width as usize) * (height as usize) * 4;
    if rgba_bytes.len() != expected_len {
        return Err(OcrError::InvalidInput {
            reason: format!(
                "RGBA buffer size mismatch: expected {} bytes ({}x{}x4), got {} bytes",
                expected_len, width, height, rgba_bytes.len()
            ),
        });
    }

    let img = image::RgbaImage::from_raw(width, height, rgba_bytes.to_vec()).ok_or_else(|| {
        OcrError::InvalidInput {
            reason: "Failed to create RgbaImage from raw bytes".to_string(),
        }
    })?;

    Ok(DynamicImage::ImageRgba8(img))
}

/// Decode a PNG/JPEG compressed byte stream into a `DynamicImage`.
pub(crate) fn decode_encoded(img_bytes: &[u8]) -> Result<DynamicImage, OcrError> {
    image::load_from_memory(img_bytes).map_err(|e| OcrError::ImageDecode {
        reason: e.to_string(),
    })
}

/// Preprocess an image for DBNet text detection.
///
/// Steps:
/// 1. Convert to RGB (drop alpha channel)
/// 2. Proportionally resize so that the longest side ≤ `max_side_len`,
///    with both dimensions rounded up to multiples of 32
/// 3. Apply ImageNet normalization: `(pixel / 255.0 - mean) / std`
/// 4. Transpose from HWC to NCHW layout
pub(crate) fn preprocess_for_detection(
    img: &DynamicImage,
    max_side_len: u32,
) -> Result<PreprocessedDetInput, OcrError> {
    let rgb = img.to_rgb8();
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    // Calculate resize dimensions (longest side ≤ max_side_len)
    let (new_w, new_h) = compute_resize_dims(orig_w, orig_h, max_side_len);

    // Resize using bilinear interpolation (Triangle filter)
    let resized = image::imageops::resize(&rgb, new_w, new_h, FilterType::Triangle);

    // Build NCHW f32 tensor with ImageNet normalization
    let tensor = rgb_to_nchw_normalized(&resized);

    Ok(PreprocessedDetInput {
        tensor,
        scale_x: new_w as f32 / orig_w as f32,
        scale_y: new_h as f32 / orig_h as f32,
        orig_width: orig_w,
        orig_height: orig_h,
        resized_rgb: resized,
    })
}

/// Preprocess a cropped text line image for CRNN recognition.
///
/// Steps:
/// 1. Resize height to `target_height` if needed, width proportionally
/// 2. Normalize RGB channels to `[-1.0, 1.0]` range: `(pixel / 255.0 - 0.5) / 0.5`
/// 3. Return as `[1, 3, H, W]` f32 tensor
#[allow(dead_code)]
pub(crate) fn preprocess_for_recognition(
    img: &RgbImage,
    target_height: u32,
) -> Array4<f32> {
    let (w, h) = (img.width(), img.height());

    let (target_width, resized) = if h == target_height {
        (w, None)
    } else {
        let aspect = w as f32 / h.max(1) as f32;
        let tw = ((target_height as f32 * aspect).round() as u32).max(1);
        let r = image::imageops::resize(img, tw, target_height, FilterType::Triangle);
        (tw, Some(r))
    };

    let mut tensor = Array4::<f32>::zeros((1, 3, target_height as usize, target_width as usize));
    for y in 0..target_height as usize {
        for x in 0..target_width as usize {
            let pixel = if let Some(ref r) = resized {
                r.get_pixel(x as u32, y as u32).0
            } else {
                img.get_pixel(x as u32, y as u32).0
            };
            tensor[[0, 0, y, x]] = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
            tensor[[0, 1, y, x]] = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
            tensor[[0, 2, y, x]] = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
        }
    }

    tensor
}

/// Fast single-pass preprocessing returning flat contiguous f32 NCHW vector directly.
pub(crate) fn preprocess_for_recognition_vec(
    img: &RgbImage,
    target_height: u32,
) -> (u32, u32, Vec<f32>) {
    let (w, h) = (img.width(), img.height());

    let (target_width, resized) = if h == target_height {
        (w, None)
    } else {
        let aspect = w as f32 / h.max(1) as f32;
        let tw = ((target_height as f32 * aspect).round() as u32).max(1);
        let r = image::imageops::resize(img, tw, target_height, FilterType::Triangle);
        (tw, Some(r))
    };

    let th = target_height as usize;
    let tw = target_width as usize;
    let plane_size = th * tw;
    let mut data = vec![0.0f32; 3 * plane_size];

    let source = resized.as_ref().unwrap_or(img);
    let raw_bytes = source.as_raw();

    for y in 0..th {
        let row_offset = y * tw;
        for x in 0..tw {
            let px_idx = (row_offset + x) * 3;
            let r = raw_bytes[px_idx];
            let g = raw_bytes[px_idx + 1];
            let b = raw_bytes[px_idx + 2];

            let dst_idx = row_offset + x;
            data[dst_idx] = (r as f32 / 255.0 - 0.5) / 0.5;
            data[plane_size + dst_idx] = (g as f32 / 255.0 - 0.5) / 0.5;
            data[2 * plane_size + dst_idx] = (b as f32 / 255.0 - 0.5) / 0.5;
        }
    }

    (target_width, target_height, data)
}

/// Pad a recognition tensor to a target width (for batching lines of different widths).
///
/// Pads with the normalization zero value (0.0 after the `/ 255.0 - 0.5` transform
/// would mean a gray pixel; we pad with -0.5 which corresponds to black).
#[allow(dead_code)]
pub(crate) fn pad_rec_tensor(
    tensor: &Array4<f32>,
    target_width: usize,
) -> Array4<f32> {
    let (_, c, h, w) = (
        tensor.shape()[0],
        tensor.shape()[1],
        tensor.shape()[2],
        tensor.shape()[3],
    );

    if w >= target_width {
        return tensor.clone();
    }

    let mut padded = Array4::<f32>::from_elem((1, c, h, target_width), -0.5);
    padded.slice_mut(ndarray::s![.., .., .., ..w]).assign(tensor);
    padded
}

/// Compute resize dimensions so that the longest side ≤ `max_side_len`
/// and both dimensions are multiples of 32 (required by DBNet stride).
fn compute_resize_dims(width: u32, height: u32, max_side_len: u32) -> (u32, u32) {
    let max_dim = width.max(height);
    let mut ratio = 1.0f32;

    if max_dim > max_side_len {
        ratio = max_side_len as f32 / max_dim as f32;
    }

    let mut new_w = (width as f32 * ratio).round() as u32;
    let mut new_h = (height as f32 * ratio).round() as u32;

    // Round up to multiples of 32
    new_w = ((new_w + 31) / 32) * 32;
    new_h = ((new_h + 31) / 32) * 32;

    // Ensure minimum size
    new_w = new_w.max(32);
    new_h = new_h.max(32);

    (new_w, new_h)
}

/// Convert an RGB image to an NCHW `[1, 3, H, W]` f32 tensor
/// with ImageNet normalization applied.
fn rgb_to_nchw_normalized(img: &RgbImage) -> Array4<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut tensor = Array4::<f32>::zeros((1, 3, h, w));

    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x as u32, y as u32).0;
            for c in 0..3 {
                tensor[[0, c, y, x]] = (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_resize_dims_no_resize() {
        let (w, h) = compute_resize_dims(640, 480, 960);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(w <= 960 && h <= 960);
    }

    #[test]
    fn test_compute_resize_dims_downscale() {
        let (w, h) = compute_resize_dims(1920, 1080, 960);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        assert!(w <= 960);
        assert!(h <= 960);
    }

    #[test]
    fn test_compute_resize_dims_small_image() {
        let (w, h) = compute_resize_dims(10, 10, 960);
        assert!(w >= 32);
        assert!(h >= 32);
    }

    #[test]
    fn test_decode_rgba_valid() {
        let width = 4u32;
        let height = 4u32;
        let rgba = vec![128u8; (width * height * 4) as usize];
        let result = decode_rgba(&rgba, width, height);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_rgba_invalid_size() {
        let result = decode_rgba(&[0u8; 10], 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_nchw_tensor_shape() {
        let img = RgbImage::from_fn(64, 32, |_, _| image::Rgb([128, 128, 128]));
        let tensor = rgb_to_nchw_normalized(&img);
        assert_eq!(tensor.shape(), &[1, 3, 32, 64]);
    }

    #[test]
    fn test_pad_rec_tensor() {
        let tensor = Array4::<f32>::zeros((1, 1, 32, 50));
        let padded = pad_rec_tensor(&tensor, 100);
        assert_eq!(padded.shape(), &[1, 1, 32, 100]);
        // Original region should be zeros
        assert_eq!(padded[[0, 0, 0, 0]], 0.0);
        // Padded region should be -0.5
        assert_eq!(padded[[0, 0, 0, 99]], -0.5);
    }
}
