//! OpenCV compatibility layer for ORT backend.
//!
//! This module provides utilities for using the ORT backend with OpenCV Mat images.
//! It enables a hybrid approach: use OpenCV for video I/O, display, and resize,
//! while using ONNX Runtime for neural network inference.
//!
//! # Features
//!
//! - Zero-copy Mat to ArrayView3 conversion (when Mat is continuous)
//! - OpenCV-based resize functions that work with BGR images
//! - Fused BGR→RGB + normalization for optimal performance
//! - `ModelTrait` for ORT models that accept OpenCV Mat input

use ndarray::{Array3, Array4, ArrayView3};
use opencv::{
    core::{Mat, MatTraitConst, Rect, Size, Scalar, BORDER_CONSTANT},
    imgproc,
    Error as OpenCvError,
};

use crate::preprocessing::{LetterboxMeta, StretchMeta, PreprocessMeta};

/// A trait for object detection models that work with OpenCV Mat.
///
/// This trait is designed for the ORT backend with OpenCV compatibility.
/// It does NOT depend on OpenCV's DNN module, avoiding static linking conflicts.
///
/// For models using the full OpenCV DNN backend, see `od_opencv::model::ModelTrait`.
pub trait ModelTrait {
    /// Run forward pass on an image and return detections.
    ///
    /// # Arguments
    /// * `image` - Input image as OpenCV Mat (BGR format)
    /// * `conf_threshold` - Confidence threshold for detections
    /// * `nms_threshold` - NMS threshold for suppressing overlapping boxes
    ///
    /// # Returns
    /// Tuple of (bounding boxes, class IDs, confidence scores)
    fn forward(
        &mut self,
        image: &Mat,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), OpenCvError>;
}

/// Converts an OpenCV Mat to an ndarray ArrayView3 (zero-copy when possible).
///
/// The Mat must be continuous in memory. If not, this function returns an error.
/// The returned view shares memory with the Mat, so the Mat must outlive the view.
///
/// # Arguments
/// * `mat` - Reference to an OpenCV Mat (expected to be BGR, CV_8UC3)
///
/// # Returns
/// * `Ok(ArrayView3<u8>)` - View over the Mat data in HWC format
/// * `Err` - If Mat is not continuous or has unexpected format
///
/// # Safety
/// The returned ArrayView borrows from the Mat's data. The Mat must not be
/// modified or dropped while the ArrayView is in use.
pub fn mat_to_array_view(mat: &Mat) -> Result<ArrayView3<'_, u8>, OpenCvError> {
    if !mat.is_continuous() {
        return Err(OpenCvError::new(
            opencv::core::StsError,
            "Mat must be continuous for zero-copy conversion",
        ));
    }

    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels() as usize;

    if channels != 3 {
        return Err(OpenCvError::new(
            opencv::core::StsError,
            format!("Expected 3 channels (BGR), got {}", channels),
        ));
    }

    let total_bytes = rows * cols * channels;
    let data_ptr = mat.data();

    let slice = unsafe { std::slice::from_raw_parts(data_ptr, total_bytes) };

    ArrayView3::from_shape((rows, cols, channels), slice).map_err(|e| {
        OpenCvError::new(
            opencv::core::StsError,
            format!("Failed to create ArrayView: {}", e),
        )
    })
}

/// Converts an OpenCV Mat to an owned ndarray Array3 (always copies).
///
/// Use this when you need the data to outlive the Mat, or when the Mat
/// is not continuous.
///
/// # Arguments
/// * `mat` - Reference to an OpenCV Mat (expected to be BGR, CV_8UC3)
///
/// # Returns
/// Owned Array3 in HWC format with BGR channel order
pub fn mat_to_array3(mat: &Mat) -> Result<Array3<u8>, OpenCvError> {
    let mat = if mat.is_continuous() {
        mat.clone()
    } else {
        let mut continuous = Mat::default();
        mat.copy_to(&mut continuous)?;
        continuous
    };

    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    let channels = mat.channels() as usize;

    let total_bytes = rows * cols * channels;
    let data_ptr = mat.data();

    let raw_data: Vec<u8> = unsafe {
        std::slice::from_raw_parts(data_ptr, total_bytes).to_vec()
    };

    Array3::from_shape_vec((rows, cols, channels), raw_data).map_err(|e| {
        OpenCvError::new(
            opencv::core::StsError,
            format!("Failed to create Array3: {}", e),
        )
    })
}

/// Resizes a BGR Mat using OpenCV, preserving BGR format.
///
/// Uses INTER_LINEAR interpolation for good balance of speed and quality.
///
/// # Arguments
/// * `mat` - Input BGR Mat
/// * `target_width` - Target width in pixels
/// * `target_height` - Target height in pixels
///
/// # Returns
/// Resized BGR Mat
pub fn resize_mat(mat: &Mat, target_width: i32, target_height: i32) -> Result<Mat, OpenCvError> {
    let mut resized = Mat::default();
    imgproc::resize(
        mat,
        &mut resized,
        Size::new(target_width, target_height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    Ok(resized)
}

/// Resizes a BGR Mat by stretching to target size, returns metadata for inverse transform.
///
/// # Arguments
/// * `mat` - Input BGR Mat
/// * `target_width` - Target width in pixels
/// * `target_height` - Target height in pixels
///
/// # Returns
/// Tuple of (resized Mat, StretchMeta)
pub fn resize_mat_stretch(
    mat: &Mat,
    target_width: i32,
    target_height: i32,
) -> Result<(Mat, StretchMeta), OpenCvError> {
    let orig_width = mat.cols();
    let orig_height = mat.rows();

    let meta = StretchMeta {
        scale_x: orig_width as f32 / target_width as f32,
        scale_y: orig_height as f32 / target_height as f32,
        original_width: orig_width,
        original_height: orig_height,
    };

    let resized = resize_mat(mat, target_width, target_height)?;
    Ok((resized, meta))
}

/// Resizes a BGR Mat with letterbox padding, returns metadata for inverse transform.
///
/// Preserves aspect ratio by scaling uniformly and padding with gray (114, 114, 114).
///
/// # Arguments
/// * `mat` - Input BGR Mat
/// * `target_width` - Target width in pixels
/// * `target_height` - Target height in pixels
///
/// # Returns
/// Tuple of (letterboxed Mat, LetterboxMeta)
pub fn resize_mat_letterbox(
    mat: &Mat,
    target_width: i32,
    target_height: i32,
) -> Result<(Mat, LetterboxMeta), OpenCvError> {
    let orig_width = mat.cols();
    let orig_height = mat.rows();

    let scale = f32::min(
        target_width as f32 / orig_width as f32,
        target_height as f32 / orig_height as f32,
    );

    let new_width = (orig_width as f32 * scale).round() as i32;
    let new_height = (orig_height as f32 * scale).round() as i32;

    let pad_left = (target_width - new_width) / 2;
    let pad_top = (target_height - new_height) / 2;

    let meta = LetterboxMeta {
        scale,
        pad_left,
        pad_top,
        original_width: orig_width,
        original_height: orig_height,
    };

    // Resize to new dimensions
    let mut resized = Mat::default();
    imgproc::resize(
        mat,
        &mut resized,
        Size::new(new_width, new_height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // Calculate padding for right and bottom
    let pad_right = target_width - new_width - pad_left;
    let pad_bottom = target_height - new_height - pad_top;

    // Add border padding with gray color (BGR: 114, 114, 114)
    let mut padded = Mat::default();
    opencv::core::copy_make_border(
        &resized,
        &mut padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        BORDER_CONSTANT,
        Scalar::new(114.0, 114.0, 114.0, 0.0),
    )?;

    Ok((padded, meta))
}

/// Full preprocessing pipeline for BGR Mat: resize + convert to tensor.
///
/// This is the optimized path that:
/// 1. Uses OpenCV for resize (works with BGR natively)
/// 2. Converts BGR→RGB fused with normalization in one pass
///
/// # Arguments
/// * `mat` - Input BGR Mat
/// * `target_width` - Target width for model input
/// * `target_height` - Target height for model input
/// * `use_letterbox` - If true, use letterbox padding; otherwise stretch
///
/// # Returns
/// Tuple of (NCHW f32 tensor in RGB, PreprocessMeta)
pub fn preprocess_mat(
    mat: &Mat,
    target_width: u32,
    target_height: u32,
    use_letterbox: bool,
) -> Result<(Array4<f32>, PreprocessMeta), OpenCvError> {
    let (resized, meta) = if use_letterbox {
        let (resized, meta) = resize_mat_letterbox(mat, target_width as i32, target_height as i32)?;
        (resized, PreprocessMeta::Letterbox(meta))
    } else {
        let (resized, meta) = resize_mat_stretch(mat, target_width as i32, target_height as i32)?;
        (resized, PreprocessMeta::Stretch(meta))
    };

    // Zero-copy view of resized Mat
    let bgr_view = mat_to_array_view(&resized)?;

    // Fused BGR→RGB + normalize in one pass
    let tensor = crate::preprocessing::bgr_hwc_to_rgb_nchw_tensor(&bgr_view);

    Ok((tensor, meta))
}

// Tests are ignored due to OpenCV library loading issues in test environment.
// These functions are tested through integration tests in the actual application.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "OpenCV tests require proper library loading - test in application"]
    fn test_mat_to_array_view() -> Result<(), OpenCvError> {
        let mat = Mat::new_rows_cols_with_default(
            2,
            2,
            opencv::core::CV_8UC3,
            Scalar::new(10.0, 20.0, 30.0, 0.0),
        )?;

        let view = mat_to_array_view(&mat)?;

        assert_eq!(view.dim(), (2, 2, 3));
        assert_eq!(view[[0, 0, 0]], 10);
        assert_eq!(view[[0, 0, 1]], 20);
        assert_eq!(view[[0, 0, 2]], 30);

        Ok(())
    }

    #[test]
    #[ignore = "OpenCV tests require proper library loading - test in application"]
    fn test_resize_mat_stretch() -> Result<(), OpenCvError> {
        let mat = Mat::new_rows_cols_with_default(
            480,
            640,
            opencv::core::CV_8UC3,
            Scalar::new(128.0, 128.0, 128.0, 0.0),
        )?;

        let (resized, meta) = resize_mat_stretch(&mat, 320, 320)?;

        assert_eq!(resized.cols(), 320);
        assert_eq!(resized.rows(), 320);
        assert!((meta.scale_x - 2.0).abs() < 0.01);
        assert!((meta.scale_y - 1.5).abs() < 0.01);

        Ok(())
    }

    #[test]
    #[ignore = "OpenCV tests require proper library loading - test in application"]
    fn test_resize_mat_letterbox() -> Result<(), OpenCvError> {
        let mat = Mat::new_rows_cols_with_default(
            480,
            640,
            opencv::core::CV_8UC3,
            Scalar::new(128.0, 128.0, 128.0, 0.0),
        )?;

        let (resized, meta) = resize_mat_letterbox(&mat, 640, 640)?;

        assert_eq!(resized.cols(), 640);
        assert_eq!(resized.rows(), 640);
        assert!(meta.scale > 0.0);
        assert!(meta.pad_top > 0 || meta.pad_left > 0);

        Ok(())
    }

    #[test]
    #[ignore = "OpenCV tests require proper library loading - test in application"]
    fn test_preprocess_mat() -> Result<(), OpenCvError> {
        let mat = Mat::new_rows_cols_with_default(
            480,
            640,
            opencv::core::CV_8UC3,
            Scalar::new(10.0, 20.0, 30.0, 0.0),
        )?;

        let (tensor, _meta) = preprocess_mat(&mat, 640, 640, true)?;

        assert_eq!(tensor.shape(), &[1, 3, 640, 640]);

        Ok(())
    }
}
