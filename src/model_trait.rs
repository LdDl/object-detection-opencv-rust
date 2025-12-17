//! Backend-agnostic object detection trait.
//!
//! This module defines the `ObjectDetector` trait that provides a common interface
//! for object detection across different inference backends (OpenCV DNN, ONNX Runtime, etc.).

use crate::BBox;

/// A trait for object detection models.
///
/// This trait provides a backend-agnostic interface for running object detection.
/// Different backends (OpenCV, ONNX Runtime, etc.) can implement this trait with
/// their own input types and error handling.
///
/// # Type Parameters
/// * `Input` - The input image type (e.g., `opencv::core::Mat`, `ImageBuffer`)
/// * `Error` - The error type for this backend
///
/// # Example
///
/// ```ignore
/// use od_opencv::{ObjectDetector, BBox};
///
/// fn run_detection<D: ObjectDetector>(
///     detector: &mut D,
///     input: &D::Input,
/// ) -> Result<Vec<BBox>, D::Error> {
///     let (bboxes, _class_ids, _confidences) = detector.detect(input, 0.5, 0.4)?;
///     Ok(bboxes)
/// }
/// ```
pub trait ObjectDetector {
    /// The input image type for this detector.
    type Input;

    /// The error type for this detector.
    type Error;

    /// Runs object detection on the input image.
    ///
    /// # Arguments
    /// * `input` - The input image
    /// * `conf_threshold` - Confidence threshold for filtering detections (0.0 to 1.0)
    /// * `nms_threshold` - Non-maximum suppression threshold (0.0 to 1.0)
    ///
    /// # Returns
    /// A tuple containing:
    /// * `Vec<BBox>` - Bounding boxes of detected objects
    /// * `Vec<usize>` - Class IDs for each detection
    /// * `Vec<f32>` - Confidence scores for each detection
    fn detect(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), Self::Error>;
}
