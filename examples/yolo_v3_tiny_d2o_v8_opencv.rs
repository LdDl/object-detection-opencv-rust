//! YOLOv3-tiny converted from Darknet via darknet2onnx (--format yolov8), OpenCV DNN backend

use std::time::Instant;

use od_opencv::{Model, DnnBackend, DnnTarget};

use opencv::{
    core::Scalar,
    imgcodecs::imread,
    imgproc::LINE_4,
    imgproc::rectangle,
};

fn main() {
    let cv_version = opencv::core::get_version_string().unwrap();
    println!("OpenCV version: {}", cv_version);

    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    let mut model = Model::opencv(
        "pretrained/yolov3-tiny-d2o-v8.onnx",
        (416, 416),
        DnnBackend::Default,
        DnnTarget::Cpu,
    ).expect("Failed to load model");

    let mut frame = imread("images/dog.jpg", 1).unwrap();

    let start = Instant::now();
    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();
    println!("Inference time: {:?}", start.elapsed());

    for (i, bbox) in bboxes.iter().enumerate() {
        rectangle(&mut frame, *bbox, Scalar::from((0.0, 255.0, 0.0)), 2, LINE_4, 0).unwrap();
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: {:?}", bbox);
        println!("\tConfidence: {:.2}", confidences[i]);
    }
}
