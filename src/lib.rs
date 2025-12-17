//! Object detection utilities for YOLO-based neural networks in OpenCV ecosystem.
//!
//! This crate provides wrappers for running YOLO models using OpenCV's DNN module.
//! Supports traditional YOLOv3, v4, v7 (Darknet), and YOLOv8, v9, v11 (Ultralytics).

// Common types
pub mod bbox;
pub use bbox::BBox;

// Backend module
pub mod backend_opencv;

// Backwards-compatible re-exports
// which allows existing code using
// `od_opencv::model_ultralytics::...` to still work
pub use backend_opencv::utils;
pub use backend_opencv::model_format;
pub use backend_opencv::model_classic;
pub use backend_opencv::model_ultralytics;
pub use backend_opencv::model;
