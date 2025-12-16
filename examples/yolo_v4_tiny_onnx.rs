use std::time::Instant;

use od_opencv::{model_classic::ModelYOLOClassic, model_format::ModelFormat};

use opencv::{
    core::{Scalar, Vector},
    dnn::DNN_BACKEND_CUDA,
    dnn::DNN_TARGET_CUDA,
    imgcodecs::imread,
    imgcodecs::imwrite,
    imgproc::rectangle,
    imgproc::LINE_4,
};

fn main() {
    let classes_labels: Vec<&str> = vec!["car", "motorbike", "bus", "truck"];
    let mf = ModelFormat::ONNX;
    let net_width = 416;
    let net_height = 416;
    // let class_filters: Vec<usize> = vec![15, 16];
    let class_filters: Vec<usize> = vec![];
    let mut model = ModelYOLOClassic::new_from_file(
        "pretrained/best_fp32.onnx",
        None,
        (net_width, net_height),
        mf,
        DNN_BACKEND_CUDA,
        DNN_TARGET_CUDA,
        vec![],
    )
    .unwrap();
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
