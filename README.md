[![Package](https://img.shields.io/crates/v/od_opencv.svg)](https://crates.io/crates/od_opencv)

# Object detection utilities in Rust programming language for YOLO-based neural networks in OpenCV ecosystem

This crate provides some basic structures and methods for solving object detections tasks via [OpenCV's DNN module](https://docs.opencv.org/4.8.0/d2/d58/tutorial_table_of_content_dnn.html). Currently implemented and tested workflows:

| Network type  | Darknet | ONNX |
| ------------- | ------------- | ------------- |
| YOLO v3 tiny  | :white_check_mark:  | :warning: (need to test)  |
| YOLO v4 tiny  | :white_check_mark:  | :warning: (need to test)  |
| YOLO v7 tiny  | :white_check_mark:   | :warning: (need to test)  |
| YOLO v3  | :white_check_mark:  | :warning: (need to test)  |
| YOLO v4  | :white_check_mark: | :warning: (need to test)  |
| YOLO v7 | :white_check_mark:  | :warning: (need to test)  |
| YOLO v8 n  | :x: (is it even possible?) | :white_check_mark:  |
| YOLO v8 s  | :x: (is it even possible?) | :white_check_mark: |
| YOLO v8 m  | :x: (is it even possible?) | :white_check_mark: |
| YOLO v8 l  | :x: (is it even possible?) | :white_check_mark: |
| YOLO v8 x  | :x: (is it even possible?) | :white_check_mark: |
| YOLO v9 t  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v9 s  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v9 m  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v9 c  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v9 e  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v11 n  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v11 s  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v11 m  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v11 l  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |
| YOLO v11 x  | :x: | :white_check_mark: (uses `ModelUltralyticsV8`) |

**Note on YOLOv9/v11:** These models use the same output format as YOLOv8 (`[1, 84, 8400]`), so `ModelUltralyticsV8` works directly. Requires OpenCV 4.10+ for best compatibility.

**Note on YOLOv10:** YOLOv10's NMS-free architecture uses TopK layer which OpenCV DNN doesn't support. Use YOLOv8/v9/v11 instead, or export YOLOv10 with [patched ultralytics](https://gist.github.com/DarthSim/216551dfd58e5628290e90c1d358704b) that removes built-in NMS.

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Features](#features)
- [References](#references)

## About

_- Why?_

Well, I just tired a bit of boilerplating (model initializing, postprocessing functions and etc.) in my both private and public projects.

_- When it is usefull?_

Well, there are several circumstances when you may need this crate:

* You need to use YOLO as your neural network base;
* You do not want use Pytorch / Tensorflow / Jax or any other DL/ML framework (someday it may happen to use pure ONNX without OpenCV features in this crate - PR's are welcome);
* You need to use OpenCV's DNN module to initialize neural network;

_- Why no YOLOv5?_

I think there is a difference in postprocessing stuff between v8 and v5 versions. I need more time to investigate what should be done exactly to make v5 work.

_- Why YOLOv10 doesn't work?_

YOLOv10 introduced an "NMS-free" architecture where post-processing (TopK selection) is built into the model itself. Unfortunately, OpenCV's DNN module doesn't support the TopK ONNX operator, causing broken inference results. You have two options:
1. Use YOLOv8/v9/v11 instead (recommended) - they work out of the box
2. Export YOLOv10 with [patched ultralytics](https://gist.github.com/DarthSim/216551dfd58e5628290e90c1d358704b) that removes built-in NMS, then use `ModelUltralyticsV8` with manual NMS

_- What OpenCV's version is tested?_

I've tested it with v4.11.0 - v4.12.0. Rust bindings version: v0.96.0

For YOLOv9 support, OpenCV 4.11+ is recommended.

_- Are wrapper structures thread safe?_

I'm not sure it is intended to be used in multiple threads (PR's are welcome). But I think you should use some queue mechanism if you want to give "async" acces to provided structs.

## Prerequisites

* For sure you must have OpenCV installed with DNN extra module. If you need to ulitize power of GPU/OpenVINO then you need to consider to include corresponding extra modules too.
    
    I love to use this [Makefile](https://github.com/hybridgroup/gocv/blob/release/Makefile) with little adjustment (OpenCV's version / enabling python bindings) for my needs.

* Prepare neural network: train it or get pretrained one. I provide pretty simple Bash scripts to download "small" versions of YOLO
    * YOLO v3 tiny - [download_v3_tiny.sh](download_v3_tiny.sh); YOLO v3 - [download_v3.sh](download_v3.sh);
    * YOLO v4 tiny - [download_v4_tiny.sh](download_v4_tiny.sh); YOLO v4 - [download_v4.sh](download_v4.sh);
    * YOLO v7 tiny - [download_v7_tiny.sh](download_v7_tiny.sh); YOLO v7 - [download_v7.sh](download_v7.sh);
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

There are some [examples](examples), but let me guide you step-by-step

1. Add this crate to your's `Cargo.toml`:
    ```shell
    cargo add od_opencv
    ```

1. Add OpenCV's bindings crate to `Cargo.toml` also:
    ```shell
    # I'm using 0.96 version
    cargo add opencv@0.96
    ```

2. Download pretrained or use your own neural network.

    I will use pretrained weights from [prerequisites section](#prerequisites)

3. Import "basic" OpenCV stuff in yours `main.rs` file:

    ```rust
    use opencv::{
        core::{Scalar, Vector},
        imgcodecs::imread,
        imgcodecs::imwrite,
        imgproc::LINE_4,
        imgproc::rectangle,
        dnn::DNN_BACKEND_CUDA, // I will utilize my GPU to perform faster inference. Your way may vary
        dnn::DNN_TARGET_CUDA,
    };
    ```
4. Import crate
    ```rust
    use od_opencv::{
        model_format::ModelFormat,
        // I'll use YOLOv8 by Ultralytics.
        // If you prefer traditional YOLO, then import it as:
        // model_classic::ModelYOLOClassic
        model_ultralytics::ModelUltralyticsV8
    };
    ```

5. Prepare model

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

6. You are good to go
    ```rust
    cargo run
    ```

7. If anything is going wrong, feel free to [open an issue](https://github.com/LdDl/object-detection-opencv-rust/issues/new)

## Features

### Letterbox Preprocessing

For non-traditional YOLO models (specifically v8 - `ModelUltralyticsV8`), you can enable letterbox preprocessing which maintains aspect ratio during resize and pads with gray borders. This matches the preprocessing used during Ultralytics training.

To enable letterbox, add the feature to your `Cargo.toml`:

```toml
[dependencies]
od_opencv = { version = "0.3", features = ["letterbox"] }
```

**Without letterbox (default):** Images are stretched to the network input size. This may introduce aspect ratio distortion.

**With letterbox:** Images are resized maintaining aspect ratio, then padded to the target size. This preserves the original aspect ratio and can be faster due to optimized buffer reuse.

## References
* YOLO v3 paper - https://arxiv.org/abs/1804.02767, Joseph Redmon, Ali Farhadi
* YOLO v4 paper - https://arxiv.org/abs/2004.10934, Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
* YOLO v7 paper - https://arxiv.org/abs/2207.02696, Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
* YOLO v9 paper - https://arxiv.org/abs/2402.13616, Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao
* YOLO v10 paper - https://arxiv.org/abs/2405.14458, Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding
* Original Darknet YOLO repository - https://github.com/pjreddie/darknet
* Most popular fork of Darknet YOLO - https://github.com/AlexeyAB/darknet
* Developers of YOLOv8/v11 - https://github.com/ultralytics/ultralytics
* Rust OpenCV's bindings - https://github.com/twistedfall/opencv-rust
* Go OpenCV's bindings (for ready-to-go Makefile) - https://github.com/hybridgroup/gocv
