//! Object detection utilities for YOLO-based neural networks.
//!
//! This crate provides wrappers for running YOLO models.
//!
//! ## Backends
//!
//! - `opencv-backend` (default): Uses OpenCV DNN for inference. Supports Darknet and ONNX models.
//! - `ort-backend`: Uses ONNX Runtime for inference. Pure Rust, no OpenCV required.
//!
//! ## Supported Models
//!
//! - Traditional YOLO (v3, v4, v7) - Darknet format, opencv-backend only
//! - Ultralytics YOLO (v8, v9, v11) - ONNX format, both backends

// Common types (always available)
pub mod bbox;
pub mod image_buffer;
pub mod model_trait;

pub use bbox::BBox;
pub use image_buffer::{ChannelOrder, ImageBuffer};
pub use model_trait::ObjectDetector;

// Pure Rust preprocessing/postprocessing (for ort-backend)
#[cfg(feature = "ort-backend")]
pub mod preprocessing;

#[cfg(feature = "ort-backend")]
pub mod postprocess;

// OpenCV backend
#[cfg(feature = "opencv-backend")]
pub mod backend_opencv;

// ONNX Runtime backend
#[cfg(feature = "ort-backend")]
pub mod backend_ort;

// Backwards-compatible re-exports for opencv-backend
// which allows existing code using
// `od_opencv::model_ultralytics::...` to still work
#[cfg(feature = "opencv-backend")]
pub use backend_opencv::utils;

#[cfg(feature = "opencv-backend")]
pub use backend_opencv::model_format;

#[cfg(feature = "opencv-backend")]
pub use backend_opencv::model_classic;

#[cfg(feature = "opencv-backend")]
pub use backend_opencv::model_ultralytics;

#[cfg(feature = "opencv-backend")]
pub use backend_opencv::model;

// Re-exports for ort-backend
#[cfg(feature = "ort-backend")]
pub use backend_ort::ModelUltralyticsOrt;

#[cfg(feature = "ort-backend")]
pub use backend_ort::OrtModelError;
