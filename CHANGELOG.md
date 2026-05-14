# Changelog

All notable changes to this set of object detection utils will be documented in this file.

## [0.10.0] - 2026-05-10
### Added
- **ArcFace face recognition** (`ModelArcFaceOrt`): 512-dim L2-normalized embedding extraction via InsightFace models
  - Supported models: `w600k_mbf` (MobileFaceNet, ~14MB) and `w600k_r50` (ResNet50, ~166MB)
  - Configurable input normalization via `ArcFaceNorm` enum: `MobileFaceNet` ([-1,1]), `ResNet` ([0,1]), `Custom { mean, scale }`
  - CPU, CUDA, TensorRT EP variants
- **Face alignment** (`face_alignment.rs`): Umeyama similarity transform (5 landmarks → affine warp with bilinear interpolation)
  - `align_face()` for default 112×112 (ArcFace recognition)
  - `align_face_sized()` for arbitrary output size (e.g. 128×128 for inswapper)
  - Exported constants: `ARCFACE_FACE_SIZE` (112), `INSWAPPER_FACE_SIZE` (128)
- **Face pipeline** (`face_pipeline.rs`): unified detect → align → embed pipeline
  - `FacePipeline`, `FaceResult`, `cosine_similarity()`
  - Factory methods: `Model::face_pipeline()`, `Model::arcface_ort()`, and `_with_norm` variants
- Examples: `arcface_ort` (MobileFaceNet), `arcface_r50_ort` (ResNet50)
- Download scripts: `download_arcface.sh`, `download_arcface_r50.sh`
- Integration tests: `test_pipeline_arnold`, `test_alignment_on_arnold`, `test_mbf_vs_r50`

---

## [0.9.0] - 2026-03-19
### Added
- **YuNet face detection** (OpenCV Zoo, 0.083M params): bounding box, 5 facial landmarks, confidence score
  - Shared decode logic in `face_detection.rs` (`FaceDetection`, `FaceDetector` trait, `nms_faces`, `decode_yunet_stride`)
  - Backend implementations: `ModelYuNetOrt` (ort-backend), `ModelYuNetRt` (tensorrt-backend), `ModelYuNetRknn` (rknn-backend), `ModelYuNetOpenCV` (opencv-backend, wrapper around `FaceDetectorYN`, requires OpenCV 4.7+)
  - Factory methods: `Model::yunet_ort()`, `Model::yunet_ort_cuda()`, `Model::yunet_ort_tensorrt()`, `Model::yunet_tensorrt()`, `Model::yunet_rknn()`, `Model::yunet_opencv()`
  - Letterbox preprocessing for YuNet (BGR [0..255], nearest-neighbor resize)
  - Input dimensions auto-detected from model metadata
- Examples: `yunet_ort`, `yunet_opencv`, `yunet_tensorrt`, `yunet_rknn`
- Developed [rv1106-yunet](https://github.com/LdDl/rv1106-yunet) — ONNX to RKNN converter for YuNet weights

---

## [0.8.2] - 2026-03-15
### Changed
- TensorRT backend no longer requires manual input size — dimensions are evaluated automatically from `.engine` file

---

## [0.8.1] - 2026-03-12
### Added
- **TensorRT + OpenCV compatibility** (`tensorrt-opencv-compat` feature): use OpenCV I/O with TensorRT inference
  - `ModelTrait` accepting `opencv::core::Mat` directly
  - Example: `yolo_v8_n_tensorrt_opencv`

---

## [0.8.0] - 2026-03-08
### Added
- **TensorRT backend** (`tensorrt-backend` feature): direct NVIDIA GPU inference via [tensorrt-infer](https://crates.io/crates/tensorrt-infer)
  - Supports TensorRT 6-8 (Jetson Nano, JetPack 4.6) and TensorRT 10+ (desktop GPUs)
  - C++ wrapper handles API differences at compile time
  - `ModelUltralyticsRt` for YOLOv8/v9/v11 `.engine` files
  - Factory methods: `Model::tensorrt()`, `Model::tensorrt_filtered()`
  - Engine files must be built from ONNX via `trtexec` on target machine (not portable across GPU architectures or TRT versions)

---

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
