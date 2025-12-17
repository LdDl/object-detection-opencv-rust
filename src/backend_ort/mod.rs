//! ONNX Runtime backend for object detection.
//!
//! This module provides YOLO model implementations using ONNX Runtime (`ort` crate).
//! It does not require OpenCV and uses pure Rust for preprocessing.

mod model_ultralytics;

pub use model_ultralytics::ModelUltralyticsOrt;
pub use model_ultralytics::OrtModelError;
