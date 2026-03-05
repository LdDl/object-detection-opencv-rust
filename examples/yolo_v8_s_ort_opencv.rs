//! YOLOv8s with ORT backend + OpenCV for image I/O
//!
//! This example demonstrates the `ort-opencv-compat` feature:
//! - OpenCV handles image loading (imread)
//! - ORT handles neural network inference
//!
//! Run with: cargo run --example yolo_v8_s_ort_opencv --features "ort-opencv-compat"

use std::time::Instant;

use opencv::{
    imgcodecs,
    imgproc,
    core::Scalar,
    prelude::*,
};
use od_opencv::{Model, ModelTrait};

fn main() {
    // Initialize ORT runtime
    ort::init().commit();

    let classes_labels: Vec<&str> = vec![
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ];

    let net_width = 640;
    let net_height = 640;

    // Create ORT model
    let mut model = Model::ort(
        "pretrained/yolov8s.onnx",
        (net_width, net_height),
    ).expect("Failed to load model");

    // Load image using OpenCV
    let mut img = imgcodecs::imread("images/dog.jpg", imgcodecs::IMREAD_COLOR)
        .expect("Failed to load image");

    println!("Image size: {}x{}", img.cols(), img.rows());

    // Run inference using ModelTrait::forward (accepts OpenCV Mat)
    // Note: We use the trait method explicitly to use Mat input
    let start = Instant::now();
    let (bboxes, class_ids, confidences) = ModelTrait::forward(&mut model, &img, 0.25, 0.4)
        .expect("Inference failed");
    println!("Inference time: {:?}", start.elapsed());

    // Draw results on image
    for (i, bbox) in bboxes.iter().enumerate() {
        let label = format!("{}: {:.2}", classes_labels[class_ids[i]], confidences[i]);

        // Draw rectangle (bbox is already opencv::core::Rect)
        imgproc::rectangle(
            &mut img,
            *bbox,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        ).unwrap();

        // Draw label
        imgproc::put_text(
            &mut img,
            &label,
            opencv::core::Point::new(bbox.x, bbox.y - 5),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        ).unwrap();

        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: x={}, y={}, w={}, h={}", bbox.x, bbox.y, bbox.width, bbox.height);
        println!("\tConfidence: {:.2}", confidences[i]);
    }

    // Save result
    imgcodecs::imwrite("output_ort_opencv.jpg", &img, &opencv::core::Vector::new())
        .expect("Failed to save image");
    println!("Result saved to output_ort_opencv.jpg");
}
