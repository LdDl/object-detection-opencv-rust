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

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
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

_- What OpenCV's version is tested?_

I've tested it with v4.7.0. Rust bindings version: v0.66.0

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

    __Notice that "v8" script downloads Pytorch-based weights and converts it into ONNX one via `ultralytics` package for Python.__
    
## Usage

There are some [examples](examples), but let me guide you step-by-step

1. Add this crate to your's `Cargo.toml`:
    ```shell
    cargo add od_opencv
    ```

1. Add OpenCV's bindings crate to `Cargo.toml` also:
    ```shell
    # I'm using 0.66 version
    cargo add opencv@0.66
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

## References
* YOLO v3 paper - https://arxiv.org/abs/1804.02767, Joseph Redmon, Ali Farhadi
* YOLO v4 paper - https://arxiv.org/abs/2004.10934, Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
* YOLO v7 paper - https://arxiv.org/abs/2207.02696, Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
* Original Darknet YOLO repository - https://github.com/pjreddie/darknet
* Most popular fork of Darknet YOLO - https://github.com/AlexeyAB/darknet
* Developers of YOLOv8 - https://github.com/ultralytics/ultralytics. If you are aware of some original papers for YOLOv8 architecture, please contact me to mention it in this README.
* Rust OpenCV's bindings - https://github.com/twistedfall/opencv-rust
* Go OpenCV's bindings (for ready-to-go Makefile) - https://github.com/hybridgroup/gocv
