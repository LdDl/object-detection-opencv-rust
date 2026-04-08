//! TensorRT backend for object detection.
//!
//! This module provides model implementations using TensorRT for NVIDIA GPUs.

mod model_ultralytics;
mod model_yunet;

pub use model_ultralytics::ModelUltralyticsRt;
pub use model_ultralytics::TrtModelError;
pub use model_yunet::ModelYuNetRt;
