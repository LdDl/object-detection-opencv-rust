# Changelog

All notable changes to this set of object detection utils will be documented in this file.

## [0.6.0] - 2026-02-07
### Added
- **RKNN NPU backend** (`rknn-backend` feature): Rockchip NPU inference via [rknn-runtime](https://github.com/LdDl/rknn-runtime) crate
  - `ModelUltralyticsRknn` for YOLOv8 models in `.rknn` format
  - Factory methods: `Model::rknn()`, `Model::rknn_filtered()`
  - Input size auto-detected from the model (no manual `input_size` parameter)
  - Custom `new_with_lib()` for non-default `librknnmrt.so` paths
  - Optimized for embedded ARM (RV1106): precomputed NC1HWC2 offsets, i8-space threshold, zero-alloc nearest-neighbor resize, lazy dequantization
  - Tested on LuckFox Pico Ultra W with COCO 320x320

---

## [0.5.0] - 2026-02-05
### Added
- YOLOv5
- YOLOv5u - "u" is for "updated". Ultralytics updated models with YOLOv8-style output, so it now easier to work with.
are supported now
## Modified
- Updated README.md with YOLOv5 support information
- Updated examples to include YOLOv5 usage
- Move download scripts to `scripts/` directory

## [0.4.1] - 2025-12-17

### Added

- **`ort-opencv-compat` feature**: Hybrid approach using ORT for inference with OpenCV for I/O
  - Enables `ModelTrait` accepting `opencv::core::Mat` directly
  - OpenCV dependency without DNN module (avoids static linking conflicts)
  - BGR HWC to RGB NCHW conversion via `preprocess_mat()`
  - New module `opencv_compat.rs` with Mat-to-ndarray utilities
---

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
