# Changelog

All notable changes to this set of object detection utils will be documented in this file.

## [0.4.0] - 2024-12-17

### Breaking Changes

- **Default backend changed**: The default feature is now `ort-backend` instead of implicit OpenCV dependency
- **Feature flags required**: OpenCV backend now requires explicit `opencv-backend` feature flag

### Added

- **ORT backend** (`ort-backend` feature): pure Rust inference using ONNX Runtime
  - `ModelUltralyticsOrt` for YOLOv8/v9/v11 models
  - No OpenCV installation required
  - CUDA support via `ort-cuda-backend` feature
  - TensorRT support via `ort-tensorrt-backend` feature
- **Pure Rust preprocessing** (`src/preprocessing.rs`): letterbox and stretch resize without OpenCV
- **Shared post-processing** (`src/postprocess.rs`): backend-agnostic NMS implementation
- **ImageBuffer type** (`src/image_buffer.rs`): common image wrapper for both backends
- **BBox type** (`src/bbox.rs`): backend-agnostic bounding box with conversions

### Migration Guide

To continue using OpenCV backend after upgrading:

```toml
# Before (0.3.x)
od_opencv = "0.3"

# After (0.4.x)
od_opencv = { version = "0.4", default-features = false, features = ["opencv-backend"] }
```

---

## [0.3.0] and earlier

OpenCV-based object detection with DNN module:

- `ModelUltralyticsV8` for YOLOv8/v9/v11 (ONNX format)
- `ModelYOLOClassic` for YOLOv3/v4/v7 (Darknet and ONNX formats)
- CUDA, OpenCL, and OpenVINO acceleration via OpenCV
- Letterbox preprocessing (optional feature) for non-traditional YOLO models
