//! OpenCV DNN backend for YOLO models.
//!
//! This module provides wrappers for running YOLO models using OpenCV's DNN module.
//! Supports YOLOv3, v4, v7 (Darknet), and YOLOv8, v9, v11 (Ultralytics).

pub mod model_format;
pub mod utils;
pub mod model;
pub mod model_classic;
pub mod model_ultralytics;

// Re-exports for convenience
pub use model_format::{ModelFormat, ModelVersion};
pub use model::ModelTrait;
pub use model_classic::ModelYOLOClassic;
pub use model_ultralytics::ModelUltralyticsV8;
