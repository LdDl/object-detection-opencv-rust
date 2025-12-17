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
pub mod model_factory;

pub use bbox::BBox;
pub use image_buffer::{ChannelOrder, ImageBuffer};
pub use model_trait::ObjectDetector;
pub use model_factory::Model;

// Pure Rust preprocessing/postprocessing (for ort-backend)
#[cfg(feature = "ort-backend")]
pub mod preprocessing;

#[cfg(feature = "ort-backend")]
pub mod postprocess;

// OpenCV DNN backend - requires opencv/dnn feature
#[cfg(feature = "opencv-backend")]
pub mod backend_opencv;

// ONNX Runtime backend
#[cfg(feature = "ort-backend")]
pub mod backend_ort;

// DNN backend/target enums for opencv-backend
#[cfg(feature = "opencv-backend")]
pub mod dnn_backend;

#[cfg(feature = "opencv-backend")]
pub use dnn_backend::{DnnBackend, DnnTarget};

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

// model module is needed for ModelTrait (opencv-backend only - depends on dnn)
#[cfg(feature = "opencv-backend")]
pub use backend_opencv::model;

// Re-exports for ort-backend
#[cfg(feature = "ort-backend")]
pub use backend_ort::ModelUltralyticsOrt;

#[cfg(feature = "ort-backend")]
pub use backend_ort::OrtModelError;

// OpenCV compatibility layer for ORT backend
// Allows using ORT inference with OpenCV Mat input
// Provides ModelTrait that does NOT depend on opencv/dnn
#[cfg(feature = "ort-opencv-compat")]
pub mod opencv_compat;

#[cfg(feature = "ort-opencv-compat")]
pub use opencv_compat::{mat_to_array_view, mat_to_array3, preprocess_mat, ModelTrait};
