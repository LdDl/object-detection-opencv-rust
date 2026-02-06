use std::time::Instant;

use od_opencv::{ImageBuffer, ModelUltralyticsRknn};

fn main() {
    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    let mut model = ModelUltralyticsRknn::new_from_file(
        "pretrained/yolov8n.rknn",
        classes_labels.len(),
        vec![],
    ).expect("Failed to load model");

    let img = image::open("images/dog.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    // Warmup
    for _ in 0..3 {
        model.forward(&img_buffer, 0.51, 0.45).expect("Warmup failed");
    }

    // Benchmark
    let iterations = 10;
    let start = Instant::now();
    let mut result = None;
    for _ in 0..iterations {
        result = Some(model.forward(&img_buffer, 0.51, 0.45).expect("Inference failed"));
    }
    let avg_ms = start.elapsed().as_millis() as f64 / iterations as f64;
    println!("{:.1} ms/inference ({:.1} FPS)", avg_ms, 1000.0 / avg_ms);

    let (bboxes, class_ids, confidences) = result.unwrap();
    println!("{} detections\n", bboxes.len());

    for (i, bbox) in bboxes.iter().enumerate() {
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: x={}, y={}, w={}, h={}", bbox.x, bbox.y, bbox.width, bbox.height);
        println!("\tConfidence: {:.2}", confidences[i]);
    }
}
