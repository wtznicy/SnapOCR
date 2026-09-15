//! C-ABI FFI exports for SnapOCR.
//!
//! Provides a stable C-compatible interface for use from non-Rust languages
//! via dynamic linking (.dll / .so / .dylib).
//!
//! # Memory Management
//! - `snap_ocr_init` allocates and returns an opaque handle.
//! - `snap_ocr_destroy` frees the handle.
//! - `snap_ocr_infer_rgba` returns a heap-allocated response.
//! - `snap_ocr_free_response` frees the response and all its strings.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::config::OcrEngineConfig;
use crate::engine::SnapOcrEngine;

/// Opaque handle to a SnapOCR engine instance.
pub type SnapOcrHandle = *mut SnapOcrEngine;

/// FFI-safe line result.
#[repr(C)]
pub struct SnapOcrLineFfi {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub x3: f32,
    pub y3: f32,
    pub score: f32,
    pub text: *mut c_char,
}

/// FFI-safe response structure.
#[repr(C)]
pub struct SnapOcrResponseFfi {
    pub full_text: *mut c_char,
    pub line_count: u32,
    pub lines: *mut SnapOcrLineFfi,
    pub total_ms: f32,
}

/// Initialize a SnapOCR engine instance.
///
/// # Arguments
/// * `model_path` — Path to the directory containing model files (null-terminated UTF-8).
/// * `num_threads` — Number of inference threads (0 for default).
///
/// # Returns
/// An opaque handle, or null on failure.
///
/// # Safety
/// `model_path` must be a valid, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn snap_ocr_init(
    model_path: *const c_char,
    num_threads: i32,
) -> SnapOcrHandle {
    if model_path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let config = OcrEngineConfig {
        model_dir: path_str.into(),
        num_threads: if num_threads > 0 {
            num_threads as usize
        } else {
            4
        },
        ..Default::default()
    };

    match SnapOcrEngine::new(config) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(e) => {
            eprintln!("SnapOCR init error: {}", e);
            ptr::null_mut()
        }
    }
}

/// Destroy a SnapOCR engine instance and free all associated resources.
///
/// # Safety
/// `handle` must be a valid handle returned by `snap_ocr_init`, or null (no-op).
/// After this call, the handle is invalid and must not be used.
#[no_mangle]
pub unsafe extern "C" fn snap_ocr_destroy(handle: SnapOcrHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Run OCR inference on raw RGBA pixel data.
///
/// # Arguments
/// * `handle` — Valid engine handle from `snap_ocr_init`.
/// * `rgba_data` — Pointer to RGBA pixel data (4 bytes per pixel).
/// * `width` — Image width in pixels.
/// * `height` — Image height in pixels.
///
/// # Returns
/// A heap-allocated response, or null on failure.
/// The caller must free the response with `snap_ocr_free_response`.
///
/// # Safety
/// - `handle` must be a valid, non-null handle.
/// - `rgba_data` must point to at least `width * height * 4` bytes.
#[no_mangle]
pub unsafe extern "C" fn snap_ocr_infer_rgba(
    handle: SnapOcrHandle,
    rgba_data: *const u8,
    width: u32,
    height: u32,
) -> *mut SnapOcrResponseFfi {
    if handle.is_null() || rgba_data.is_null() {
        return ptr::null_mut();
    }

    let engine = &*handle;
    let data_len = (width as usize) * (height as usize) * 4;
    let rgba_slice = std::slice::from_raw_parts(rgba_data, data_len);

    match engine.recognize_rgba(rgba_slice, width, height) {
        Ok(result) => {
            // Convert OcrResult to FFI response
            let full_text = CString::new(result.text.as_str())
                .unwrap_or_default()
                .into_raw();

            let line_count = result.lines.len() as u32;

            let lines = if result.lines.is_empty() {
                ptr::null_mut()
            } else {
                let mut ffi_lines: Vec<SnapOcrLineFfi> = result
                    .lines
                    .iter()
                    .map(|line| {
                        let text = CString::new(line.text.as_str())
                            .unwrap_or_default()
                            .into_raw();
                        SnapOcrLineFfi {
                            x0: line.box_coords.points[0][0],
                            y0: line.box_coords.points[0][1],
                            x1: line.box_coords.points[1][0],
                            y1: line.box_coords.points[1][1],
                            x2: line.box_coords.points[2][0],
                            y2: line.box_coords.points[2][1],
                            x3: line.box_coords.points[3][0],
                            y3: line.box_coords.points[3][1],
                            score: line.score,
                            text,
                        }
                    })
                    .collect();

                let ptr = ffi_lines.as_mut_ptr();
                std::mem::forget(ffi_lines);
                ptr
            };

            let response = Box::new(SnapOcrResponseFfi {
                full_text,
                line_count,
                lines,
                total_ms: result.latency_ms.total_ms,
            });

            Box::into_raw(response)
        }
        Err(e) => {
            eprintln!("SnapOCR inference error: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free a response returned by `snap_ocr_infer_rgba`.
///
/// # Safety
/// `response` must be a valid pointer returned by `snap_ocr_infer_rgba`, or null (no-op).
/// After this call, the response pointer is invalid.
#[no_mangle]
pub unsafe extern "C" fn snap_ocr_free_response(response: *mut SnapOcrResponseFfi) {
    if response.is_null() {
        return;
    }

    let resp = Box::from_raw(response);

    // Free the full_text string
    if !resp.full_text.is_null() {
        drop(CString::from_raw(resp.full_text));
    }

    // Free each line's text string and the lines array
    if !resp.lines.is_null() && resp.line_count > 0 {
        let lines =
            Vec::from_raw_parts(resp.lines, resp.line_count as usize, resp.line_count as usize);
        for line in &lines {
            if !line.text.is_null() {
                drop(CString::from_raw(line.text as *mut c_char));
            }
        }
    }
}
