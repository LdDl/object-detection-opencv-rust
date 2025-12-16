use std::time::Instant;

use od_opencv::{
    model_format::ModelFormat,
    model_ultralytics::ModelUltralyticsV8
};

use opencv::{
    core::{Scalar, Vector},
    imgcodecs::imread,
    imgcodecs::imwrite,
    imgproc::LINE_4,
    imgproc::rectangle,
    dnn::DNN_BACKEND_CUDA,
    dnn::DNN_TARGET_CUDA,
};

fn main() {
    // Print OpenCV version
    let cv_version = opencv::core::get_version_string().unwrap();
    println!("OpenCV version: {}", cv_version);

    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
    let mf = ModelFormat::ONNX;
    let net_width = 640;
    let net_height = 640;
    // let class_filters: Vec<usize> = vec![15, 16]; // filter specific classes
    let class_filters: Vec<usize> = vec![]; // empty = all classes

    // YOLOv9 uses the same output format as YOLOv8: [1, 84, 8400]
    // So we can use ModelUltralyticsV8 directly
    let mut model = ModelUltralyticsV8::new_from_file(
        "pretrained/yolov9s.onnx",
        None,
        (net_width, net_height),
        mf,
        DNN_BACKEND_CUDA,
        DNN_TARGET_CUDA,
        class_filters
    ).unwrap();

    let mut frame = imread("images/dog.jpg", 1).unwrap();
    let start = Instant::now();
    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();
    println!("Inference time: {:?}", start.elapsed());

    for (i, bbox) in bboxes.iter().enumerate() {
        rectangle(&mut frame, *bbox, Scalar::from((0.0, 255.0, 0.0)), 2, LINE_4, 0).unwrap();
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: {:?}", bbox);
        println!("\tConfidences: {}", confidences[i]);
    }

    imwrite("images/dog_yolov9_s.jpg", &frame, &Vector::new()).unwrap();
}
