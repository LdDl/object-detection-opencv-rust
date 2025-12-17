# Changelog

All notable changes to this set of object detection utils will be documented in this file.

## [0.4.0] - 2024-12-17

### Breaking Changes

- **Default backend changed**: The default feature is now `ort-backend` instead of implicit OpenCV dependency
- **Feature flags required**: OpenCV backend now requires explicit `opencv-backend` feature flag

### Added

- **Factory pattern API** (`Model` struct): simplified model instantiation
  - `Model::opencv()` for Ultralytics models (YOLOv8/v9/v11) with OpenCV backend
  - `Model::darknet()` for traditional YOLO (v3/v4/v7) in Darknet format
  - `Model::classic_onnx()` for classic YOLO models exported to ONNX
  - `Model::ort()` for Ultralytics models with ORT backend
  - `Model::ort_cuda()` for ORT backend with CUDA acceleration
- **DnnBackend and DnnTarget enums** (`src/dnn_backend.rs`): type-safe OpenCV DNN configuration
  - Import from `od_opencv` instead of `opencv::dnn`
  - Available backends: `Default`, `OpenCV`, `InferenceEngine`, `Halide`, `Cuda`
  - Available targets: `Cpu`, `OpenCL`, `OpenCLFp16`, `Myriad`, `Fpga`, `Cuda`, `CudaFp16`, `Hddl`
- **ORT backend** (`ort-backend` feature): pure Rust inference using ONNX Runtime
  - `ModelUltralyticsOrt` for YOLOv8/v9/v11 models
  - No OpenCV installation required
  - CUDA support via `ort-cuda-backend` feature
  - TensorRT support via `ort-tensorrt-backend` feature
- **Pure Rust preprocessing** (`src/preprocessing.rs`): letterbox and stretch resize without OpenCV
- **Shared post-processing** (`src/postprocess.rs`): backend-agnostic NMS implementation
- **ImageBuffer type** (`src/image_buffer.rs`): common image wrapper for both backends
- **BBox type** (`src/bbox.rs`): backend-agnostic bounding box with conversions

### Important Notes

- **CUDA conflict warning**: Do not enable both ORT and OpenCV backends simultaneously when using CUDA. Always use `default-features = false` when enabling `opencv-backend` to avoid segmentation faults.

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
