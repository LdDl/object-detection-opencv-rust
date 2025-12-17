use std::time::Instant;

use od_opencv::{Model, DnnBackend, DnnTarget};

use opencv::{
    core::{Scalar, Vector},
    imgcodecs::imread,
    imgcodecs::imwrite,
    imgproc::rectangle,
    imgproc::LINE_4,
};

fn main() {
    // Print OpenCV version
    let cv_version = opencv::core::get_version_string().unwrap();
    println!("OpenCV version: {}", cv_version);

    let classes_labels: Vec<&str> = vec!["car", "motorbike", "bus", "truck"];

    let net_width = 416;
    let net_height = 416;

    let mut model = Model::classic_onnx("pretrained/best_fp32.onnx", (net_width, net_height), DnnBackend::Cuda, DnnTarget::Cuda).unwrap();
    let mut frame = imread(
        "images/dog.jpg",
        1,
    )
    .unwrap();
    let start = Instant::now();
    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.75, 0.4).unwrap();
    println!("Inference time: {:?}", start.elapsed());
    for (i, bbox) in bboxes.iter().enumerate() {
        rectangle(
            &mut frame,
            *bbox,
            Scalar::from((0.0, 255.0, 0.0)),
            2,
            LINE_4,
            0,
        )
        .unwrap();
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: {:?}", bbox);
        println!("\tConfidences: {}", confidences[i]);
    }
    imwrite("images/dog_yolov4_tiny.jpg", &frame, &Vector::new()).unwrap();
}
