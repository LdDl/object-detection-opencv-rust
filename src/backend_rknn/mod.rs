//! RKNN NPU backend for object detection.
//!
//! This module provides YOLO model implementations using the RKNN NPU runtime
//! for Rockchip devices.

mod model_ultralytics;

pub use model_ultralytics::ModelUltralyticsRknn;
pub use model_ultralytics::RknnModelError;
