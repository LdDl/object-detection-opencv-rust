use std::time::Instant;

use od_opencv::{Model, DnnBackend, DnnTarget};

use opencv::{
    core::{Point, Scalar, Vector},
    imgcodecs::imread,
    imgcodecs::imwrite,
    imgproc::{LINE_4, circle, rectangle},
};

fn main() {
    let mut model = Model::yunet_opencv(
        "pretrained/face_detection_yunet_2023mar.onnx",
        (320, 320),
        DnnBackend::OpenCV,
        DnnTarget::Cpu,
    ).expect("Failed to load YuNet model");

    println!("Input size: {:?}", model.input_size());

    let mut frame = imread("images/oscar_selfies.jpg", 1).expect("Failed to load image");

    // Warmup
    let _ = model.forward(&frame, 0.7, 0.3);

    let start = Instant::now();
    let detections = model.forward(&frame, 0.7, 0.3).expect("Inference failed");
    println!("Inference time: {:?}", start.elapsed());
    println!("Found {} face(s)", detections.len());

    let bbox_color = Scalar::from((0.0, 255.0, 0.0));
    let landmark_colors = [
        Scalar::from((0.0, 0.0, 255.0)),     // right_eye - red (BGR)
        Scalar::from((255.0, 0.0, 0.0)),     // left_eye - blue
        Scalar::from((0.0, 255.0, 255.0)),   // nose - yellow
        Scalar::from((255.0, 0.0, 255.0)),   // right_mouth - magenta
        Scalar::from((255.0, 255.0, 0.0)),   // left_mouth - cyan
    ];

    for (i, det) in detections.iter().enumerate() {
        println!("Face #{}: confidence={:.3}, bbox=({:.1},{:.1},{:.1},{:.1})",
            i + 1, det.confidence, det.x, det.y, det.width, det.height);
        let names = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"];
        for (j, name) in names.iter().enumerate() {
            println!("  {}: ({:.1}, {:.1})", name, det.landmarks[j][0], det.landmarks[j][1]);
        }

        let rect = opencv::core::Rect::new(
            det.x as i32, det.y as i32,
            det.width as i32, det.height as i32,
        );
        rectangle(&mut frame, rect, bbox_color, 2, LINE_4, 0).unwrap();

        for (j, &color) in landmark_colors.iter().enumerate() {
            let pt = Point::new(det.landmarks[j][0] as i32, det.landmarks[j][1] as i32);
            circle(&mut frame, pt, 3, color, -1, LINE_4, 0).unwrap();
        }
    }

    let output_path = "images/oscar_selfies_yunet_opencv.jpg";
    imwrite(output_path, &frame, &Vector::new()).expect("Failed to save output");
    println!("Saved to {}", output_path);
}
