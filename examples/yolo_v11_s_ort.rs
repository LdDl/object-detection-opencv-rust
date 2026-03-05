use std::time::Instant;

use od_opencv::{ImageBuffer, Model};

fn main() {
    // Initialize ort runtime
    ort::init().commit();

    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    let net_width = 640;
    let net_height = 640;

    let mut model = Model::ort(
        "pretrained/yolo11s.onnx",
        (net_width, net_height),
    ).expect("Failed to load model");

    let img = image::open("images/dog.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    let start = Instant::now();
    let (bboxes, class_ids, confidences) = model.forward(&img_buffer, 0.25, 0.4).expect("Inference failed");
    println!("Inference time: {:?}", start.elapsed());

    for (i, bbox) in bboxes.iter().enumerate() {
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: x={}, y={}, w={}, h={}", bbox.x, bbox.y, bbox.width, bbox.height);
        println!("\tConfidence: {:.2}", confidences[i]);
    }
}
