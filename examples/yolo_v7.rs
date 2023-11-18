use od_opencv::{
    model_format::ModelFormat,
    model_classic::ModelYOLOClassic
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
    let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
    let mf = ModelFormat::Darknet;
    let net_width = 416;
    let net_height = 416;
    // let class_filters: Vec<usize> = vec![15, 16];
    let class_filters: Vec<usize> = vec![];
    let mut model = ModelYOLOClassic::new_from_file("pretrained/yolov7.weights", Some("pretrained/yolov7.cfg"), (net_width, net_height), mf, DNN_BACKEND_CUDA, DNN_TARGET_CUDA, class_filters).unwrap();
    let mut frame = imread("images/dog.jpg", 1).unwrap();
    let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();
    for (i, bbox) in bboxes.iter().enumerate() {
        rectangle(&mut frame, *bbox, Scalar::from((0.0, 255.0, 0.0)), 2, LINE_4, 0).unwrap();
        println!("Class: {}", classes_labels[class_ids[i]]);
        println!("\tBounding box: {:?}", bbox);
        println!("\tConfidences: {}", confidences[i]);
    }
    imwrite("images/dog_yolov7.jpg", &frame, &Vector::new()).unwrap();
}
