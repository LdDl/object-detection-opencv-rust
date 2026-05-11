use std::time::Instant;

use image::{Rgb, RgbImage};
use od_opencv::{ImageBuffer, Model, FaceDetection};

fn main() {
    let mut model = Model::yunet_tensorrt(
        "pretrained/face_detection_yunet_2023mar.engine",
    ).expect("Failed to load YuNet engine");

    println!("Input size: {:?}", model.input_size());

    let img = image::open("images/oscar_selfies.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img.clone());

    // Warmup
    let _ = model.forward(&img_buffer, 0.7, 0.3);

    let start = Instant::now();
    let detections = model.forward(&img_buffer, 0.7, 0.3).expect("Inference failed");
    println!("Inference time: {:?}", start.elapsed());
    println!("Found {} face(s)", detections.len());

    // Draw results
    let mut canvas = img.to_rgb8();

    let bbox_color = Rgb([0u8, 255, 0]);
    let landmark_colors = [
        Rgb([255u8, 0, 0]),     // right_eye - red
        Rgb([0u8, 0, 255]),     // left_eye - blue
        Rgb([255u8, 255, 0]),   // nose - yellow
        Rgb([255u8, 0, 255]),   // right_mouth - magenta
        Rgb([0u8, 255, 255]),   // left_mouth - cyan
    ];

    for (i, det) in detections.iter().enumerate() {
        println!("Face #{}: confidence={:.3}, bbox=({:.1},{:.1},{:.1},{:.1})",
            i + 1, det.confidence, det.x, det.y, det.width, det.height);
        let names = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"];
        for (j, name) in names.iter().enumerate() {
            println!("  {}: ({:.1}, {:.1})", name, det.landmarks[j][0], det.landmarks[j][1]);
        }

        draw_rect(&mut canvas, det, bbox_color);

        for (j, &color) in landmark_colors.iter().enumerate() {
            let lx = det.landmarks[j][0] as i32;
            let ly = det.landmarks[j][1] as i32;
            draw_circle(&mut canvas, lx, ly, 3, color);
        }
    }

    let output_path = "images/oscar_selfies_yunet_tensorrt.jpg";
    canvas.save(output_path).expect("Failed to save output");
    println!("Saved to {}", output_path);
}

fn draw_rect(img: &mut RgbImage, det: &FaceDetection, color: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let x0 = (det.x as i32).clamp(0, w - 1);
    let y0 = (det.y as i32).clamp(0, h - 1);
    let x1 = ((det.x + det.width) as i32).clamp(0, w - 1);
    let y1 = ((det.y + det.height) as i32).clamp(0, h - 1);

    // Top & bottom
    for x in x0..=x1 {
        img.put_pixel(x as u32, y0 as u32, color);
        img.put_pixel(x as u32, y1 as u32, color);
    }
    // Left & right
    for y in y0..=y1 {
        img.put_pixel(x0 as u32, y as u32, color);
        img.put_pixel(x1 as u32, y as u32, color);
    }
}

fn draw_circle(img: &mut RgbImage, cx: i32, cy: i32, r: i32, color: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }
}
