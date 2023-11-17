# Object detection utilities in Rust programming language for YOLO-based neural networks in OpenCV ecosystem

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
- [Usage](#usage)
- [References](#references)

## About
@todo

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


