use std::time::Instant;

use od_opencv::{
    ImageBuffer,
    backend_ort::ModelUltralyticsOrt,
};

fn main() {
    // Initialize ort runtime
    ort::init().commit().expect("Failed to initialize ORT");

    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

    let net_width = 640;
    let net_height = 640;

    // Load model using ORT backend with CUDA
    println!("Loading model with CUDA...");
    let mut model = ModelUltralyticsOrt::new_from_file_cuda(
        "pretrained/yolov8s.onnx",
        (net_width, net_height),
        vec![],  // no class filters
    ).expect("Failed to load model");

    // Load image using the image crate
    println!("Loading image...");
    let img = image::open("images/dog.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img);

    // Warmup run (CUDA initialization)
    println!("Warmup run...");
    let _ = model.forward(&img_buffer, 0.25, 0.4).expect("Warmup failed");

    // Run inference multiple times
    println!("Running inference (10 iterations)...");
    let start = Instant::now();
    let mut result = None;
    for _ in 0..10 {
        result = Some(model.forward(&img_buffer, 0.25, 0.4).expect("Inference failed"));
    }
    let elapsed = start.elapsed();
    println!("Total time: {:?}", elapsed);
    println!("Average per frame: {:?}", elapsed / 10);

    let (bboxes, class_ids, confidences) = result.unwrap();

    // Print results
    println!("\nDetections:");
    for (i, bbox) in bboxes.iter().enumerate() {
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: x={}, y={}, w={}, h={}", bbox.x, bbox.y, bbox.width, bbox.height);
        println!("\tConfidence: {:.2}", confidences[i]);
    }

    println!("\nTotal detections: {}", bboxes.len());
}
