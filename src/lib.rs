//! Object detection utilities for YOLO-based neural networks.
//!
//! This crate provides wrappers for running YOLO models. Currently supports
//! the OpenCV DNN backend via the `opencv-backend` feature.
//!
//! Supports traditional YOLOv3, v4, v7 (Darknet), and YOLOv8, v9, v11 (Ultralytics).

// Common types (always available)
pub mod bbox;
pub mod image_buffer;
pub mod model_trait;

pub use bbox::BBox;
pub use image_buffer::{ChannelOrder, ImageBuffer};
pub use model_trait::ObjectDetector;

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
