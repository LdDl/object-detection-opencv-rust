[![Crates.io](https://img.shields.io/crates/v/od_opencv.svg)](https://crates.io/crates/od_opencv)
[![Downloads](https://img.shields.io/crates/d/od_opencv.svg)](https://crates.io/crates/od_opencv)
[![Documentation](https://docs.rs/od_opencv/badge.svg)](https://docs.rs/od_opencv)

# Object detection utilities in Rust for YOLO-based neural networks

This crate provides structures and methods for solving object detection tasks using YOLO models. It supports multiple inference backends:

Also this crate provides face detection via YuNet model.

- **ORT backend** (default): Pure Rust, no OpenCV required, uses ONNX Runtime
- **OpenCV backend**: Uses [OpenCV's DNN module](https://docs.opencv.org/4.8.0/d2/d58/tutorial_table_of_content_dnn.html), supports Darknet format
- **TensorRT backend**: Direct TensorRT inference via [tensorrt-infer](https://crates.io/crates/tensorrt-infer), for NVIDIA GPUs and Jetson devices
- **RKNN backend**: Rockchip NPU inference for edge devices (RV1106 tested only on LuckFox Pico Ultra W)

| Network type  | ORT (ONNX) | OpenCV (ONNX) | OpenCV (Darknet) | TensorRT (.engine) |
| ------------- | ---------- | ------------- | ---------------- | ------------------ |
| YOLO v3 tiny  | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: | :white_check_mark: (via [darknet2onnx] + trtexec) |
| YOLO v4 tiny  | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: | :white_check_mark: (via [darknet2onnx] + trtexec) |
| YOLO v7 tiny  | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: | :white_check_mark: (via [darknet2onnx] + trtexec) |
| YOLO v3       | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: | :white_check_mark: (via [darknet2onnx] + trtexec) |
| YOLO v4       | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: | :white_check_mark: (via [darknet2onnx] + trtexec) |
| YOLO v7       | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: (via [darknet2onnx]) | :white_check_mark: | :white_check_mark: (via [darknet2onnx] + trtexec) |
| YOLO v5u n/s/m/l/x | :white_check_mark: (uses `Model::ort()`) | :white_check_mark: (uses `Model::opencv()`) | :x: | :white_check_mark: (uses `Model::tensorrt()`) |
| YOLO v5 n/s/m/l/x  | :white_check_mark: (uses `Model::yolov5_ort()`) | :white_check_mark: (uses `Model::yolov5_opencv()`) | :x: | :x: |
| YOLO v8 n/s/m/l/x | :white_check_mark: | :white_check_mark: | :x: (is it even possible?) | :white_check_mark: (uses `Model::tensorrt()`) |
| YOLO v9 t/s/m/c/e | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: | :white_check_mark: (uses `Model::tensorrt()`) |
| YOLO v11 n/s/m/l/x | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: | :white_check_mark: (uses `Model::tensorrt()`) |
| **Face Detection** | | | | |
| YuNet (OpenCV Zoo) | :white_check_mark: (uses `Model::yunet_ort()`) | :white_check_mark: (uses `Model::yunet_opencv()`) | :x: | :white_check_mark: (uses `Model::yunet_tensorrt()`) |

**Note on YOLOv3/v4/v7 ONNX:** Darknet `.cfg` + `.weights` can be converted to ONNX using [darknet2onnx](https://github.com/LdDl/darknet2onnx). Use `--format yolov8` to get `[1, 84, N]` output compatible with `Model::ort()` / `Model::opencv()` / `Model::tensorrt()` or `--format yolov5` for `[1, N, 85]` compatible with `Model::yolov5_ort()` / `Model::yolov5_opencv()`. E.g. using `yolov8` format:
```bash
darknet2onnx --cfg yolov4-tiny.cfg --weights yolov4-tiny.weights --output yolov4-tiny-d2o-v8.onnx --format yolov8
```

For TensorRT inference you should prepare `.engine` file:
```bash
trtexec --onnx=yolov4-tiny-d2o-v8.onnx --saveEngine=yolov4-tiny-d2o-v8.engine --fp16
```
Also be aware: I've tested only `yolov8` format for TensorRT.

[darknet2onnx]: https://github.com/LdDl/darknet2onnx

**Note on YOLOv9/v11:** These models use the same output format as YOLOv8 (`[1, 84, 8400]`), so `ModelUltralyticsV8` works directly. For opencv-backend it is required to use OpenCV v4.11+ for best compatibility.

**Note on YOLOv10:** YOLOv10's NMS-free architecture uses TopK layer which OpenCV DNN doesn't support. Use YOLOv8/v9/v11 instead, or export YOLOv10 with [patched ultralytics](https://gist.github.com/DarthSim/216551dfd58e5628290e90c1d358704b) that removes built-in NMS. For ORT backend I've not tested YOLOv10 yet.

## Table of Contents

- [About](#about)
- [Backends](#backends)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [ORT Backend (Default)](#ort-backend-default)
  - [OpenCV Backend](#opencv-backend)
  - [TensorRT Backend](#tensorrt-backend)
  - [RKNN Backend](#rknn-backend)
  - [Face Detection (YuNet)](#face-detection-yunet)
- [Features](#features)
- [Migration from 0.3.x](#migration-from-03x)
- [References](#references)

## About

_- Why?_

Well, I just tired a bit of boilerplating (model initializing, postprocessing functions and etc.) in my both private and public projects.

_- When it is usefull?_

Well, there are several circumstances when you may need this crate:

* You need to use YOLO as your neural network base;
* You do not want use Pytorch / Tensorflow / Jax or any other DL/ML framework (someday it may happen to use pure ONNX without OpenCV features in this crate - PR's are welcome);
* You need to use OpenCV's DNN module to initialize neural network? Use `opencv-backend` feature;

* You want a simple Rust-native solution without heavy dependencies? Use `ort-backend` feature;

_- What about YOLOv5?_

YOLOv5 is supported (05.02.2026 update)! There are two variants:

1. **YOLOv5u** (yolov5nu, yolov5su, etc.) - Ultralytics updated models with YOLOv8-style output (`[1, 84, 8400]`). Use `Model::ort()` or `Model::opencv()` - same as YOLOv8.
2. **Original YOLOv5** (yolov5n, yolov5s, etc.) - Classic format with objectness score (`[1, 25200, 85]`). Use `Model::yolov5_ort()` or `Model::yolov5_opencv()`.

Download scripts:
- YOLOv5u: `scripts/download_v5u_n.sh`, `scripts/download_v5u_s.sh`, etc.
- Original: `scripts/download_v5_n.sh`, `scripts/download_v5_s.sh`, etc.

_- Why YOLOv10 doesn't work with OpenCV?_

YOLOv10 introduced an "NMS-free" architecture where post-processing (TopK selection) is built into the model itself. Unfortunately, OpenCV's DNN module doesn't support the TopK ONNX operator, causing broken inference results. You have two options:
1. Use YOLOv8/v9/v11 instead (recommended) - they work out of the box
2. Export YOLOv10 with [patched ultralytics](https://gist.github.com/DarthSim/216551dfd58e5628290e90c1d358704b) that removes built-in NMS, then use `ModelUltralyticsV8` with manual NMS

_- What OpenCV's version is tested for `opencv-backend`?_

I've tested it with v4.11.0 - v4.12.0. Rust bindings version: v0.96.0

For YOLOv9/v11 support, OpenCV 4.11+ is recommended.

_- Are wrapper structures thread safe?_

I'm not sure it is intended to be used in multiple threads (PR's are welcome). But I think you should use some queue mechanism if you want to give "async" acces to provided structs.

## Backends

This crate supports multiple inference backends:

| Backend | Default | OpenCV Required | GPU Support | Models Supported |
|---------|---------|-----------------|-------------|------------------|
| `ort-backend` | Yes | No | CUDA, TensorRT | YOLOv5/v5u/v8/v9/v11 (ONNX), YuNet face detection |
| `opencv-backend` | No | Yes | CUDA, OpenCL, OpenVINO | All YOLO versions, YuNet face detection |
| `tensorrt-backend` | No | No | NVIDIA GPU (TensorRT) | YOLOv8/v9/v11 (.engine), YuNet face detection |
| `rknn-backend` | No | No | Rockchip NPU | YOLOv8/v9/v11 (.rknn), YuNet face detection |

**Warning: CUDA Conflict**

Do not enable both `ort-backend` (or `ort-cuda-backend`) and `opencv-backend` simultaneously when using CUDA acceleration. The CUDA initialization from ORT and OpenCV can conflict, causing segmentation faults at runtime.

Always use `default-features = false` when enabling `opencv-backend` to disable the default ORT backend.

### Choosing a Backend

```toml
# Use ORT backend (default) - no OpenCV installation needed
od_opencv = "0.6"

# Use ORT backend with CUDA acceleration
od_opencv = { version = "0.6", features = ["ort-cuda-backend"] }

# Use ORT backend with TensorRT acceleration. WARNING: I DID NOT TESTED IT MYSELF!
od_opencv = { version = "0.6", features = ["ort-tensorrt-backend"] }

# Use OpenCV backend (required for Darknet models)
od_opencv = { version = "0.6", default-features = false, features = ["opencv-backend"] }

# Use TensorRT backend (NVIDIA GPUs, Jetson devices)
od_opencv = { version = "0.8", default-features = false, features = ["tensorrt-backend"] }

# Use RKNN backend (Rockchip NPU devices)
od_opencv = { version = "0.8", default-features = false, features = ["rknn-backend"] }
```

## Prerequisites

### For ORT Backend (default)

No special system dependencies required. The `ort` crate will download ONNX Runtime automatically.

For GPU acceleration:
- **CUDA**: Install CUDA toolkit and cuDNN
- **TensorRT**: Install TensorRT (includes CUDA requirement)

### For OpenCV Backend

For sure you must have OpenCV installed with DNN extra module. If you need to ulitize power of GPU/OpenVINO then you need to consider to include corresponding extra modules too.

I love to use this [Makefile](https://github.com/hybridgroup/gocv/blob/release/Makefile) with little adjustment (OpenCV's version / enabling python bindings) for my needs.

Tested with OpenCV v4.11.0 - v4.12.0. Rust bindings version: v0.96.0

### For TensorRT Backend

- NVIDIA GPU with CUDA support
- CUDA toolkit with `libcudart`
- TensorRT (6.x, 8.x, or 10.x) with `libnvinfer`
- C++ compiler supporting C++14

Engine files must be built from ONNX using `trtexec` on the target machine. Engine files are **not portable** between GPU architectures or TensorRT versions.

```bash
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

Tested platforms:

| Platform | TensorRT | CUDA | Architecture |
|---|---|---|---|
| Jetson Nano (JetPack 4.6) | 8.2 | 10.2 | aarch64 |
| Desktop (RTX 3060, Arch Linux) | 10.15 | 12.x | x86_64 |

For non-standard CUDA/TensorRT installation paths, set environment variables `CUDA_HOME`, `CUDA_LIB_DIR`, `TENSORRT_LIB_DIR`. See [tensorrt-infer-sys](https://crates.io/crates/tensorrt-infer-sys) for details.

### Getting Models

* Prepare neural network: train it or get pretrained one. I provide pretty simple Bash scripts to download some versions of YOLO (located in [scripts/](scripts/) folder):
    * Traditional: YOLO v3 tiny - [download_v3_tiny.sh](scripts/download_v3_tiny.sh); YOLO v3 - [download_v3.sh](scripts/download_v3.sh);
    * Traditional: YOLO v4 tiny - [download_v4_tiny.sh](scripts/download_v4_tiny.sh); YOLO v4 - [download_v4.sh](scripts/download_v4.sh);
    * Traditional: YOLO v7 tiny - [download_v7_tiny.sh](scripts/download_v7_tiny.sh); YOLO v7 - [download_v7.sh](scripts/download_v7.sh);
    * YOLO v5u (Ultralytics updated, YOLOv8 format): [download_v5u_n.sh](scripts/download_v5u_n.sh), [download_v5u_s.sh](scripts/download_v5u_s.sh), [download_v5u_m.sh](scripts/download_v5u_m.sh), [download_v5u_l.sh](scripts/download_v5u_l.sh), [download_v5u_x.sh](scripts/download_v5u_x.sh).
    * YOLO v5 (original format with objectness): [download_v5_n.sh](scripts/download_v5_n.sh), [download_v5_s.sh](scripts/download_v5_s.sh), [download_v5_m.sh](scripts/download_v5_m.sh), [download_v5_l.sh](scripts/download_v5_l.sh), [download_v5_x.sh](scripts/download_v5_x.sh).
    * YOLO v8 nano (n) - [download_v8_n.sh](scripts/download_v8_n.sh).
    * YOLO v8 small (s) - [download_v8_s.sh](scripts/download_v8_s.sh).
    * YOLO v8 medium (m) - [download_v8_m.sh](scripts/download_v8_m.sh).
    * YOLO v8 large (l) - [download_v8_l.sh](scripts/download_v8_l.sh).
    * YOLO v8 extra (x) - [download_v8_x.sh](scripts/download_v8_x.sh).
    * YOLO v9 tiny (t) - [download_v9_t.sh](scripts/download_v9_t.sh).
    * YOLO v9 small (s) - [download_v9_s.sh](scripts/download_v9_s.sh).
    * YOLO v9 medium (m) - [download_v9_m.sh](scripts/download_v9_m.sh).
    * YOLO v9 compact (c) - [download_v9_c.sh](scripts/download_v9_c.sh).
    * YOLO v9 extended (e) - [download_v9_e.sh](scripts/download_v9_e.sh).
    * YOLO v11 nano (n) - [download_v11_n.sh](scripts/download_v11_n.sh).
    * YOLO v11 small (s) - [download_v11_s.sh](scripts/download_v11_s.sh).
    * YOLO v11 medium (m) - [download_v11_m.sh](scripts/download_v11_m.sh).
    * YOLO v11 large (l) - [download_v11_l.sh](scripts/download_v11_l.sh).
    * YOLO v11 extra (x) - [download_v11_x.sh](scripts/download_v11_x.sh).

    __Notice that "v5/v5u/v8/v9/v11" scripts download Pytorch-based weights and convert them into ONNX via `ultralytics` package for Python.__
    
## Usage

There are some [examples](examples), but let me guide you step-by-step.

### Running Examples

```bash
# ORT backend examples (default)
cargo run --example yolo_v8_n_ort --release
cargo run --example yolo_v8_n_ort_cuda --release --features=ort-cuda-backend

# OpenCV backend examples - IMPORTANT: use --no-default-features to avoid CUDA conflicts
cargo run --example yolo_v4_tiny --release --no-default-features --features=opencv-backend
cargo run --example yolo_v8_n --release --no-default-features --features=opencv-backend

# Darknet-to-ONNX examples (names contain "d2o") - converted via darknet2onnx
cargo run --example yolo_v4_tiny_d2o_v8_ort --release
cargo run --example yolo_v4_tiny_d2o_v8_opencv --release --no-default-features --features=opencv-backend

# Face detection (YuNet)
cargo run --example yunet_ort --release
cargo run --example yunet_opencv --release --no-default-features --features=opencv-backend
cargo run --example yunet_tensorrt --release --no-default-features --features=tensorrt-backend
```

> **Note:** Examples with `d2o` in the name use ONNX models converted from Darknet `.cfg` + `.weights` via [darknet2onnx]. The suffix `v8` or `v5` indicates the output format used during conversion (`--format yolov8` or `--format yolov5`).

> **Note:** When running OpenCV backend examples, always use `--no-default-features` to disable the ORT backend. Without this flag, both backends will be loaded which can cause segmentation faults when using CUDA

### ORT Backend (Default)

The ORT backend uses ONNX Runtime and doesn't require OpenCV. This is the recommended approach for YOLOv8/v9/v11 models.

1. Add this crate to your's `Cargo.toml`:
    ```shell
    cargo add od_opencv
    ```

2. Download pretrained or use your own neural network.

    I will use pretrained weights from [prerequisites section](#getting-models)

3. Import crate in yours `main.rs` file:

    ```rust
    use od_opencv::{
        ImageBuffer,
        backend_ort::ModelUltralyticsOrt,
    };
    ```

4. Prepare model

    ```rust
    // Initialize ORT runtime
    ort::init().commit().expect("Failed to initialize ORT");

    // Define classes (in this case we consider 80 COCO labels)
    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    // Define model's input size
    let net_width = 640;
    let net_height = 640;

    // Initialize optional filters.
    // E.g.: if you do want to find only dogs and cats and you can't re-train neural network,
    // then you can just place `vec![15, 16]` to filter dogs and cats (15 - index of `cat` in class labels, 16 - `dog`)
    // let class_filters: Vec<usize> = vec![15, 16];
    let class_filters: Vec<usize> = vec![];

    // Initialize model itself
    let mut model = ModelUltralyticsOrt::new_from_file(
        "pretrained/yolov8s.onnx",
        (net_width, net_height),
        class_filters,
    ).expect("Failed to load model");

    // Load image using the `image` crate
    let img = image::open("images/dog.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    // Feed forward image through the model
    let (bboxes, class_ids, confidences) = model.forward(
        &img_buffer,
        0.25,  // confidence threshold
        0.4,   // NMS threshold
    ).expect("Inference failed");

    // Process results
    for (i, bbox) in bboxes.iter().enumerate() {
        // Debug output to stdin
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBBox: x={}, y={}, w={}, h={}", bbox.x, bbox.y, bbox.width, bbox.height);
        println!("\tConfidence: {:.2}", confidences[i]);
    }
    ```

5. You are good to go
    ```shell
    cargo run
    ```

6. If anything is going wrong, feel free to [open an issue](https://github.com/LdDl/object-detection-opencv-rust/issues/new)

#### ORT with CUDA

If you want to use CUDA acceleration, change your `Cargo.toml`:

```toml
[dependencies]
od_opencv = { version = "0.6", features = ["ort-cuda-backend"] }
```

And use `new_from_file_cuda` instead of `new_from_file`:

```rust
let mut model = ModelUltralyticsOrt::new_from_file_cuda(
    "pretrained/yolov8s.onnx",
    (net_width, net_height),
    class_filters,
).expect("Failed to load model");
```

#### ORT with OpenCV I/O

If you need OpenCV for video capture or image I/O but want ORT for inference, use the `ort-opencv-compat` feature:

```toml
od_opencv = { version = "0.6", features = ["ort-opencv-compat"] }
```

This enables `ModelTrait` which accepts `opencv::core::Mat` directly:

```rust
use od_opencv::{Model, ModelTrait};
use opencv::imgcodecs;

let mut model = Model::ort("model.onnx", (640, 640))?;
let img = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;
let (bboxes, class_ids, confidences) = ModelTrait::forward(&mut model, &img, 0.25, 0.4)?;
```

See full example here: [examples/yolo_v8_s_ort_opencv.rs](examples/yolo_v8_s_ort_opencv.rs)

### OpenCV Backend

The OpenCV backend is required for Darknet models (v3/v4/v7) and provides access to highly optimized OpenCV's things for preprocessing and postprocessing.

1. Add this crate to your's `Cargo.toml`:
    ```shell
    cargo add od_opencv --no-default-features --features opencv-backend
    ```

2. Add OpenCV's bindings crate to `Cargo.toml` also:
    ```shell
    # I'm using 0.96 version
    cargo add opencv@0.96
    ```

3. Download pretrained or use your own neural network.

    I will use pretrained weights from [prerequisites section](#getting-models)

4. Import "basic" OpenCV stuff in yours `main.rs` file:

    ```rust
    use opencv::{
        core::{Scalar, Vector},
        imgcodecs::imread,
        imgcodecs::imwrite,
        imgproc::LINE_4,
        imgproc::rectangle,
    };
    ```

5. Import crate (choose one approach):

    **Option A: Factory pattern (recommended)**
    ```rust
    use od_opencv::{
        Model,
        DnnBackend, // I will utilize my GPU to perform faster inference. Your way may vary
        DnnTarget
    };
    ```

    **Option B: Direct struct access**
    ```rust
    use od_opencv::{
        model_format::ModelFormat,
        // I'll use YOLOv8 by Ultralytics.
        // If you prefer traditional YOLO, then import it as:
        // model_classic::ModelYOLOClassic
        model_ultralytics::ModelUltralyticsV8
    };
    use opencv::dnn::{
        DNN_BACKEND_CUDA, // I will utilize my GPU to perform faster inference. Your way may vary
        DNN_TARGET_CUDA
    };
    ```

6. Prepare model

    **Option A: Factory pattern (recommended)**
    ```rust
    // Define classes (in this case we consider 80 COCO labels)
    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    // Initialize model using factory pattern
    let mut model = Model::opencv(
        "pretrained/yolov8n.onnx",
        (640, 640),
        DnnBackend::Cuda,
        DnnTarget::Cuda,
    ).unwrap();

    // Read image into the OpenCV's Mat object
    let mut frame = imread("images/dog.jpg", 1).unwrap();

    // Feed forward image through the model
    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();

    // Process results
    for (i, bbox) in bboxes.iter().enumerate() {
        rectangle(&mut frame, *bbox, Scalar::from((0.0, 255.0, 0.0)), 2, LINE_4, 0).unwrap();
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: {:?}", bbox);
        println!("\tConfidences: {}", confidences[i]);
    }

    imwrite("images/dog_yolov8_n.jpg", &frame, &Vector::new()).unwrap();
    ```

    **Option B: Direct struct access**
    ```rust
    // Define classes (in this case we consider 80 COCO labels)
    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    // Define format for OpenCV's DNN module
    let mf = ModelFormat::ONNX;

    // Define model's input size
    let net_width = 640;
    let net_height = 640;

    // Initialize optional filters.
    // E.g.: if you do want to find only dogs and cats and you can't re-train neural network,
    // then you can just place `vec![15, 16]` to filter dogs and cats (15 - index of `cat` in class labels, 16 - `dog`)
    // let class_filters: Vec<usize> = vec![15, 16];
    let class_filters: Vec<usize> = vec![];

    // Initialize model itself
    let mut model = ModelUltralyticsV8::new_from_file("pretrained/yolov8n.onnx", None, (net_width, net_height), mf, DNN_BACKEND_CUDA, DNN_TARGET_CUDA, class_filters).unwrap();

    // Read image into the OpenCV's Mat object
    // Define it as mutable since we are going to put bounding boxes onto it.
    let mut frame = imread("images/dog.jpg", 1).unwrap();

    // Feed forward image through the model
    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();

    // Process results
    for (i, bbox) in bboxes.iter().enumerate() {
        // Place bounding boxes onto the image
        rectangle(&mut frame, *bbox, Scalar::from((0.0, 255.0, 0.0)), 2, LINE_4, 0).unwrap();
        // Debug output to stdin
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: {:?}", bbox);
        println!("\tConfidences: {}", confidences[i]);
    }

    // Finally save the updated image to the file system
    imwrite("images/dog_yolov8_n.jpg", &frame, &Vector::new()).unwrap();
    ```

7. You are good to go
    ```shell
    cargo run --no-default-features --features=opencv-backend
    ```

8. If anything is going wrong, feel free to [open an issue](https://github.com/LdDl/object-detection-opencv-rust/issues/new)

### TensorRT Backend

The TensorRT backend runs inference directly on NVIDIA GPUs via [tensorrt-infer](https://crates.io/crates/tensorrt-infer). It supports TensorRT 6-8 (Jetson Nano with JetPack 4.6) and TensorRT 10+ (desktop GPUs). The C++ wrapper handles API differences at compile time.

1. Build an engine file from ONNX on the target machine:
    ```bash
    trtexec --onnx=pretrained/yolov8n.onnx --saveEngine=pretrained/yolov8n.engine --fp16
    ```

2. Add to `Cargo.toml`:
    ```toml
    [dependencies]
    od_opencv = { version = "0.8", default-features = false, features = ["tensorrt-backend"] }
    image = "0.25"
    ```

3. Use the model:

    **Option A: Factory pattern**
    ```rust
    use od_opencv::{ImageBuffer, Model};

    let mut model = Model::tensorrt("pretrained/yolov8n.engine")
        .expect("Failed to load engine");

    let img = image::open("images/dog.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    let (bboxes, class_ids, confidences) = model.forward(&img_buffer, 0.25, 0.4)
        .expect("Inference failed");
    ```

    **Option B: Direct struct access**
    ```rust
    use od_opencv::{ImageBuffer, backend_tensorrt::ModelUltralyticsRt};

    let mut model = ModelUltralyticsRt::new_from_file(
        "pretrained/yolov8n.engine",
        vec![],   // class_filters (empty = all classes)
    ).expect("Failed to load engine");

    let img = image::open("images/dog.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    let (bboxes, class_ids, confidences) = model.forward(&img_buffer, 0.25, 0.4)
        .expect("Inference failed");
    ```

4. Run:
    ```bash
    cargo run --example yolo_v8_n_tensorrt --no-default-features --features tensorrt-backend --release
    ```

**Note:** Engine files are tied to the specific GPU architecture and TensorRT version. You must rebuild the `.engine` file on each target machine. Input dimensions are read directly from the engine bindings, so there is no need to pass them manually.

#### TensorRT with OpenCV I/O

If you need OpenCV for video capture or image I/O but want TensorRT for inference, use the `tensorrt-opencv-compat` feature:

```toml
od_opencv = { version = "0.8", default-features = false, features = ["tensorrt-opencv-compat"] }
```

This enables `ModelTrait` which accepts `opencv::core::Mat` directly:

```rust
use od_opencv::{Model, ModelTrait};
use opencv::imgcodecs;

let mut model = Model::tensorrt("pretrained/yolov8n.engine")?;
let img = imgcodecs::imread("image.jpg", imgcodecs::IMREAD_COLOR)?;
let (bboxes, class_ids, confidences) = ModelTrait::forward(&mut model, &img, 0.25, 0.4)?;
```

See full example here: [examples/yolo_v8_n_tensorrt_opencv.rs](examples/yolo_v8_n_tensorrt_opencv.rs)

### RKNN Backend

The RKNN backend runs inference on Rockchip NPU using the [rknn-runtime](https://github.com/LdDl/rknn-runtime) crate. Tested on LuckFox Pico Ultra W (RV1106) with a COCO 320x320 model. For that specific size I've converted ONNX model to `.rknn` (with some preparations also) format via recommendations here: [rv1106-yolov8](https://github.com/LdDl/rv1106-yolov8).

1. Add to `Cargo.toml`:
    ```toml
    [dependencies]
    od_opencv = { version = "0.6", default-features = false, features = ["rknn-backend"] }
    image = "0.25"
    ```

    **Note:** Use `default-features = false` to avoid pulling in ORT dependencies during cross-compilation.

2. Use the model:

    **Option A: Factory pattern**
    ```rust
    use od_opencv::{ImageBuffer, Model};

    let mut model = Model::rknn("yolov8n.rknn", 80).expect("Failed to load model");

    let img = image::open("image.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    let (bboxes, class_ids, confidences) = model.forward(&img_buffer, 0.51, 0.45)
        .expect("Inference failed");
    ```

    **Option B: Direct struct access**
    ```rust
    use od_opencv::{ImageBuffer, ModelUltralyticsRknn};

    let mut model = ModelUltralyticsRknn::new_from_file(
        "yolov8n.rknn",
        80,       // num_classes
        vec![],   // class_filters (empty = all classes)
    ).expect("Failed to load model");

    let img = image::open("image.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    let (bboxes, class_ids, confidences) = model.forward(&img_buffer, 0.51, 0.45)
        .expect("Inference failed");
    ```

    Input size is auto-detected from the model (no need to specify it manually).

    If `librknnmrt.so` is not at the default path (`/usr/lib/librknnmrt.so`), use `new_with_lib`:
    ```rust
    let mut model = ModelUltralyticsRknn::new_with_lib(
        "yolov8n.rknn",
        "/opt/rknn/lib/librknnmrt.so",
        80,
        vec![],
    ).expect("Failed to load model");
    ```

    **Note on confidence threshold:** INT8 quantization causes `sigmoid(0) = 0.5` to dequantize to ~0.502. Use a threshold > 0.502 (e.g. `0.51` literally) to avoid false positives from zero-initialized outputs.

    **Warning:** The RKNN backend uses `unsafe` code extensively - raw pointer arithmetic for nearest-neighbor resize, `get_unchecked` for NC1HWC2 tensor access, and FFI calls to `librknnmrt.so` via `rknn-runtime`. Use at your own risk.

3. Cross-compile and deploy:
    ```bash
    cross build --target armv7-unknown-linux-gnueabihf --release \
        --example yolo_v8_n_rknn --no-default-features --features rknn-backend
    scp target/armv7-unknown-linux-gnueabihf/release/examples/yolo_v8_n_rknn user@device:~/
    ```

### Face Detection (YuNet)

This crate supports face detection using [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) from OpenCV Zoo. YuNet is an extremely lightweight model (0.083M params, 228KB ONNX) that detects faces and returns 5 facial landmarks (eyes, nose, mouth corners).

The model is available for ORT, OpenCV, TensorRT, and RKNN backends. For ORT/TensorRT/RKNN, input dimensions are read from the model automatically. For OpenCV, the built-in `FaceDetectorYN` handles all preprocessing and decoding internally.

Download the ONNX model from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet):
```bash
wget -O pretrained/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

For TensorRT, convert the ONNX model:
```bash
trtexec --onnx=pretrained/face_detection_yunet_2023mar.onnx \
  --saveEngine=pretrained/face_detection_yunet_2023mar.engine --fp16
```

**Usage (ORT backend):**
```rust
use od_opencv::{ImageBuffer, Model, FaceDetection};

ort::init().commit();

let mut model = Model::yunet_ort("pretrained/face_detection_yunet_2023mar.onnx")
    .expect("Failed to load YuNet model");

let img = image::open("image.jpg").expect("Failed to load image");
let img_buffer = ImageBuffer::from_dynamic_image(img);

let detections = model.forward(&img_buffer, 0.7, 0.3)
    .expect("Inference failed");

for det in &detections {
    println!("Face: confidence={:.3}, bbox=({:.1},{:.1},{:.1},{:.1})",
        det.confidence, det.x, det.y, det.width, det.height);
    let names = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"];
    for (j, name) in names.iter().enumerate() {
        println!("  {}: ({:.1}, {:.1})", name, det.landmarks[j][0], det.landmarks[j][1]);
    }
}
```

**Usage (TensorRT backend):**
```rust
use od_opencv::{ImageBuffer, Model};

let mut model = Model::yunet_tensorrt("pretrained/face_detection_yunet_2023mar.engine")
    .expect("Failed to load engine");

let img = image::open("image.jpg").expect("Failed to load image");
let img_buffer = ImageBuffer::from_dynamic_image(img);

let detections = model.forward(&img_buffer, 0.7, 0.3)
    .expect("Inference failed");
```

**Usage (OpenCV backend):**
```rust
use od_opencv::{Model, DnnBackend, DnnTarget};

let mut model = Model::yunet_opencv(
    "pretrained/face_detection_yunet_2023mar.onnx",
    (320, 320),
    DnnBackend::OpenCV,
    DnnTarget::Cpu,
)?;

let frame = opencv::imgcodecs::imread("image.jpg", 1)?;
let detections = model.forward(&frame, 0.7, 0.3)?;
```

See full examples: [examples/yunet_ort.rs](examples/yunet_ort.rs), [examples/yunet_opencv.rs](examples/yunet_opencv.rs), [examples/yunet_tensorrt.rs](examples/yunet_tensorrt.rs)

## Features

### Letterbox Preprocessing

For non-traditional YOLO models (v8/v9/v11), you can enable letterbox preprocessing which maintains aspect ratio during resize and pads with gray borders. This matches the preprocessing used during Ultralytics training.

This works for ORT, OpenCV, and TensorRT backends.

```toml
# ORT backend with letterbox
od_opencv = { version = "0.6", features = ["letterbox"] }

# OpenCV backend with letterbox
od_opencv = { version = "0.6", default-features = false, features = ["opencv-backend", "letterbox"] }
```

**Without letterbox (default):** Images are stretched to the network input size. This may introduce aspect ratio distortion.

**With letterbox:** Images are resized maintaining aspect ratio, then padded to the target size. This preserves the original aspect ratio and can be faster due to optimized buffer reuse.

## Migration from 0.3.x

In version 0.4.0, the default backend changed from OpenCV to ORT:

| Version | Default Backend | OpenCV Required |
|---------|-----------------|-----------------|
| 0.3.x   | opencv (implicit) | Yes |
| 0.4.x   | ort-backend | No |

**To keep using OpenCV backend after upgrading:**

```toml
# Before (0.3.x)
od_opencv = "0.3"

# After (0.4.x) - explicitly enable opencv-backend
od_opencv = { version = "0.4", default-features = false, features = ["opencv-backend"] }
```

Your existing code using `ModelUltralyticsV8`, `ModelYOLOClassic`, etc. will continue to work unchanged with the `opencv-backend` feature.

## References
* YOLO v3 paper - https://arxiv.org/abs/1804.02767, Joseph Redmon, Ali Farhadi
* YOLO v4 paper - https://arxiv.org/abs/2004.10934, Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
* YOLO v7 paper - https://arxiv.org/abs/2207.02696, Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
* YOLO v9 paper - https://arxiv.org/abs/2402.13616, Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao
* YOLO v10 paper - https://arxiv.org/abs/2405.14458, Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding
* YuNet face detection paper - https://link.springer.com/article/10.1007/s11633-023-1423-y, Shiqi Yu, Yuanbo Xia, et al.
* YuNet training repository (libfacedetection) - https://github.com/ShiqiYu/libfacedetection.train
* YuNet pretrained weights (OpenCV Zoo) - https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
* Original Darknet YOLO repository - https://github.com/pjreddie/darknet
* Most popular fork of Darknet YOLO - https://github.com/AlexeyAB/darknet
* Developers of YOLOv8/v11 - https://github.com/ultralytics/ultralytics
* ONNX Runtime - https://onnxruntime.ai/

* Darknet to ONNX converter (Go CLI) - https://github.com/LdDl/darknet2onnx
* Rust OpenCV's bindings - https://github.com/twistedfall/opencv-rust
* Go OpenCV's bindings (for ready-to-go Makefile) - https://github.com/hybridgroup/gocv
* RKNN Runtime (Rust bindings for Rockchip NPU) - https://github.com/LdDl/rknn-runtime
* ONNX to RKNN conversion for RV1106 - https://github.com/LdDl/rv1106-yolov8
* TensorRT safe Rust wrappers - https://github.com/LdDl/tensorrt-infer
* TensorRT FFI bindings - https://github.com/LdDl/tensorrt-infer-sys
