[![Package](https://img.shields.io/crates/v/od_opencv.svg)](https://crates.io/crates/od_opencv)

# Object detection utilities in Rust for YOLO-based neural networks

This crate provides structures and methods for solving object detection tasks using YOLO models. It supports multiple inference backends:

- **ORT backend** (default): Pure Rust, no OpenCV required, uses ONNX Runtime
- **OpenCV backend**: Uses [OpenCV's DNN module](https://docs.opencv.org/4.8.0/d2/d58/tutorial_table_of_content_dnn.html), supports Darknet format

| Network type  | ORT (ONNX) | OpenCV (ONNX) | OpenCV (Darknet) |
| ------------- | ---------- | ------------- | ---------------- |
| YOLO v3 tiny  | :x:        | :warning: (need to test) | :white_check_mark: |
| YOLO v4 tiny  | :x:        | :warning: (need to test) | :white_check_mark: |
| YOLO v7 tiny  | :x:        | :warning: (need to test) | :white_check_mark: |
| YOLO v3       | :x:        | :warning: (need to test) | :white_check_mark: |
| YOLO v4       | :x:        | :warning: (need to test) | :white_check_mark: |
| YOLO v7       | :x:        | :warning: (need to test) | :white_check_mark: |
| YOLO v8 n     | :white_check_mark: | :white_check_mark: | :x: (is it even possible?) |
| YOLO v8 s     | :white_check_mark: | :white_check_mark: | :x: (is it even possible?) |
| YOLO v8 m     | :white_check_mark: | :white_check_mark: | :x: (is it even possible?) |
| YOLO v8 l     | :white_check_mark: | :white_check_mark: | :x: (is it even possible?) |
| YOLO v8 x     | :white_check_mark: | :white_check_mark: | :x: (is it even possible?) |
| YOLO v9 t     | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v9 s     | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v9 m     | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v9 c     | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v9 e     | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v11 n    | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v11 s    | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v11 m    | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v11 l    | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |
| YOLO v11 x    | :white_check_mark: (uses `ModelUltralyticsOrt`) | :white_check_mark: (uses `ModelUltralyticsV8`) | :x: |

**Note on YOLOv9/v11:** These models use the same output format as YOLOv8 (`[1, 84, 8400]`), so `ModelUltralyticsV8` works directly. For opencv-backend it is required to use OpenCV v4.11+ for best compatibility.

**Note on YOLOv10:** YOLOv10's NMS-free architecture uses TopK layer which OpenCV DNN doesn't support. Use YOLOv8/v9/v11 instead, or export YOLOv10 with [patched ultralytics](https://gist.github.com/DarthSim/216551dfd58e5628290e90c1d358704b) that removes built-in NMS. For ORT backend I've not tested YOLOv10 yet.

## Table of Contents

- [About](#about)
- [Backends](#backends)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [ORT Backend (Default)](#ort-backend-default)
  - [OpenCV Backend](#opencv-backend)
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

_- Why no YOLOv5?_

I think there is a difference in postprocessing stuff between v8 and v5 versions. I need more time to investigate what should be done exactly to make v5 work.

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

This crate supports two inference backends:

| Backend | Default | OpenCV Required | GPU Support | Models Supported |
|---------|---------|-----------------|-------------|------------------|
| `ort-backend` | Yes | No | CUDA, TensorRT | YOLOv8/v9/v11 (ONNX) |
| `opencv-backend` | No | Yes | CUDA, OpenCL, OpenVINO | All YOLO versions |

**Warning: CUDA Conflict**

Do not enable both `ort-backend` (or `ort-cuda-backend`) and `opencv-backend` simultaneously when using CUDA acceleration. The CUDA initialization from ORT and OpenCV can conflict, causing segmentation faults at runtime.

Always use `default-features = false` when enabling `opencv-backend` to disable the default ORT backend.

### Choosing a Backend

```toml
# Use ORT backend (default) - no OpenCV installation needed
od_opencv = "0.4"

# Use ORT backend with CUDA acceleration
od_opencv = { version = "0.4", features = ["ort-cuda-backend"] }

# Use ORT backend with TensorRT acceleration. WARNING: I DID NOT TESTED IT MYSELF!
od_opencv = { version = "0.4", features = ["ort-tensorrt-backend"] }

# Use OpenCV backend (required for Darknet models)
od_opencv = { version = "0.4", default-features = false, features = ["opencv-backend"] }
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

### Getting Models

* Prepare neural network: train it or get pretrained one. I provide pretty simple Bash scripts to download some versions of YOLO
    * Traditional: YOLO v3 tiny - [download_v3_tiny.sh](download_v3_tiny.sh); YOLO v3 - [download_v3.sh](download_v3.sh);
    * Traditional: YOLO v4 tiny - [download_v4_tiny.sh](download_v4_tiny.sh); YOLO v4 - [download_v4.sh](download_v4.sh);
    * Traditional: YOLO v7 tiny - [download_v7_tiny.sh](download_v7_tiny.sh); YOLO v7 - [download_v7.sh](download_v7.sh);
    * YOLO v8 nano (n) - [download_v8_n.sh](download_v8_n.sh).
    * YOLO v8 small (s) - [download_v8_s.sh](download_v8_s.sh).
    * YOLO v8 medium (m) - [download_v8_m.sh](download_v8_m.sh).
    * YOLO v8 large (l) - [download_v8_l.sh](download_v8_l.sh).
    * YOLO v8 extra (x) - [download_v8_x.sh](download_v8_x.sh).
    * YOLO v9 tiny (t) - [download_v9_t.sh](download_v9_t.sh).
    * YOLO v9 small (s) - [download_v9_s.sh](download_v9_s.sh).
    * YOLO v9 medium (m) - [download_v9_m.sh](download_v9_m.sh).
    * YOLO v9 compact (c) - [download_v9_c.sh](download_v9_c.sh).
    * YOLO v9 extended (e) - [download_v9_e.sh](download_v9_e.sh).
    * YOLO v11 nano (n) - [download_v11_n.sh](download_v11_n.sh).
    * YOLO v11 small (s) - [download_v11_s.sh](download_v11_s.sh).
    * YOLO v11 medium (m) - [download_v11_m.sh](download_v11_m.sh).
    * YOLO v11 large (l) - [download_v11_l.sh](download_v11_l.sh).
    * YOLO v11 extra (x) - [download_v11_x.sh](download_v11_x.sh).

    __Notice that "v8/v9/v11" scripts download Pytorch-based weights and convert them into ONNX via `ultralytics` package for Python.__
    
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
```

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
od_opencv = { version = "0.4", features = ["ort-cuda-backend"] }
```

And use `new_from_file_cuda` instead of `new_from_file`:

```rust
let mut model = ModelUltralyticsOrt::new_from_file_cuda(
    "pretrained/yolov8s.onnx",
    (net_width, net_height),
    class_filters,
).expect("Failed to load model");
```

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

## Features

### Letterbox Preprocessing

For non-traditional YOLO models (v8/v9/v11), you can enable letterbox preprocessing which maintains aspect ratio during resize and pads with gray borders. This matches the preprocessing used during Ultralytics training.

This works for both ORT and OpenCV backends.

```toml
# ORT backend with letterbox
od_opencv = { version = "0.4", features = ["letterbox"] }

# OpenCV backend with letterbox
od_opencv = { version = "0.4", default-features = false, features = ["opencv-backend", "letterbox"] }
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
* Original Darknet YOLO repository - https://github.com/pjreddie/darknet
* Most popular fork of Darknet YOLO - https://github.com/AlexeyAB/darknet
* Developers of YOLOv8/v11 - https://github.com/ultralytics/ultralytics
* ONNX Runtime - https://onnxruntime.ai/

* Rust OpenCV's bindings - https://github.com/twistedfall/opencv-rust
* Go OpenCV's bindings (for ready-to-go Makefile) - https://github.com/hybridgroup/gocv
