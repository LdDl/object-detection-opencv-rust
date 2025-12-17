//! Object detection utilities for YOLO-based neural networks.
//!
//! This crate provides wrappers for running YOLO models. Currently supports
//! the OpenCV DNN backend via the `opencv-backend` feature.
//!
//! Supports traditional YOLOv3, v4, v7 (Darknet), and YOLOv8, v9, v11 (Ultralytics).

// Common types
pub mod bbox;
pub use bbox::BBox;

// Backend module (for opencv-backend feature)
#[cfg(feature = "opencv-backend")]
pub mod backend_opencv;

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
