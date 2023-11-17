# Object detection utilities in Rust programming language for YOLO-based neural networks in OpenCV ecosystem

## W.I.P - There is no crate currently (2023.11.17)
## Crate will be published to crates.io when I write more tests and docs

This crate provides some basic structures and methods for solving object detections tasks via [OpenCV's DNN module](https://docs.opencv.org/4.8.0/d2/d58/tutorial_table_of_content_dnn.html). Currently implemented and tested workflows:

| Network type  | Darknet | ONNX |
| ------------- | ------------- | ------------- |
| YOLO v3 tiny  | :white_check_mark:  | :x:  |
| YOLO v4 tiny  | :question: (need to test) | :x:  |
| YOLO v7 tiny  | :question: (need to test)  | :x:  |
| YOLO v3  | :question: (need to test)  | :x:  |
| YOLO v4  | :question: (need to test)  | :x:  |
| YOLO v7 | :question: (need to test)  | :x:  |
| YOLO v8 n  | :x: (is it even possible?) | :white_check_mark:  |

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [References](#references)

## About
@todo

## Prerequisites

* For sure you must have OpenCV installed with DNN extra module. If you need to ulitize power of GPU/OpenVINO then you need to consider corresponding extra modules too.
    
    I love to use this [Makefile](https://github.com/hybridgroup/gocv/blob/release/Makefile) with little adjustment (OpenCV's version / enabling python bindings) for my needs.

* Prepare neural network: train it or get pretrained one. I provide pretty simple Bash scripts to download "small" versions of YOLO
    * YOLO v3 tiny - [download_v3_tiny.sh](download_v3_tiny.sh);
    * YOLO v4 tiny - [download_v4_tiny.sh](download_v4_tiny.sh);
    * YOLO v7 tiny - [download_v7_tiny.sh](download_v7_tiny.sh);
    * YOLO v8 nano (n) - [download_v8_n.sh](download_v8_n.sh). Notice that this script downloads Pytorch-based weights and converts it into ONNX one via `ultralytics` package for Python.
    
## Usage
@todo

## References
* YOLO v3 paper - https://arxiv.org/abs/1804.02767, Joseph Redmon, Ali Farhadi
* YOLO v4 paper - https://arxiv.org/abs/2004.10934, Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
* YOLO v7 paper - https://arxiv.org/abs/2207.02696, Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
* Original Darknet YOLO repository - https://github.com/pjreddie/darknet
* Most popular fork of Darknet YOLO - https://github.com/AlexeyAB/darknet
* Developers of YOLOv8 - https://github.com/ultralytics/ultralytics. If you are aware of some original papers for YOLOv8 architecture, please contact me to mention it in this README.
* Rust OpenCV's bindings - https://github.com/twistedfall/opencv-rust
* Go OpenCV's bindings (for ready-to-go Makefile) - https://github.com/hybridgroup/gocv
