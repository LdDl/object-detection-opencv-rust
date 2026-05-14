use std::time::Instant;

use image::{Rgb, RgbImage};
use od_opencv::{ImageBuffer, Model, FaceResult, cosine_similarity};

fn main() {
    ort::init().commit();

    let mut pipeline = Model::face_pipeline(
        "pretrained/face_detection_yunet_2023mar.onnx",
        "pretrained/w600k_mbf.onnx",
    ).expect("Failed to load face pipeline");

    println!("Detector input size: {:?}", pipeline.input_size());

    let img = image::open("images/arnold.jpg").expect("Failed to load image");
    let img_buffer = ImageBuffer::from_dynamic_image(img.clone());

    // Warmup
    let _ = pipeline.process(&img_buffer, 0.7, 0.3);

    let start = Instant::now();
    let faces = pipeline.process(&img_buffer, 0.7, 0.3).expect("Pipeline failed");
    println!("Pipeline time: {:?}", start.elapsed());
    println!("Found {} face(s)", faces.len());

    // Print face info
    for (i, face) in faces.iter().enumerate() {
        let norm: f32 = face.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("Face #{}: confidence={:.3}, bbox=({:.1},{:.1},{:.1},{:.1}), embedding L2 norm={:.4}",
            i + 1, face.confidence, face.x, face.y, face.width, face.height, norm);
    }

    // If multiple faces, print pairwise similarities
    if faces.len() >= 2 {
        println!("\nPairwise cosine similarities:");
        for i in 0..faces.len() {
            for j in (i + 1)..faces.len() {
                let sim = cosine_similarity(&faces[i].embedding, &faces[j].embedding);
                println!("  Face #{} vs Face #{}: {:.4}", i + 1, j + 1, sim);
            }
        }
    }

    // Draw results
    let mut canvas = img.to_rgb8();
    let bbox_color = Rgb([0u8, 255, 0]);
    let landmark_colors = [
        Rgb([255u8, 0, 0]),     // left_eye - red
        Rgb([0u8, 0, 255]),     // right_eye - blue
        Rgb([255u8, 255, 0]),   // nose - yellow
        Rgb([255u8, 0, 255]),   // left_mouth - magenta
        Rgb([0u8, 255, 255]),   // right_mouth - cyan
    ];

    for face in &faces {
        draw_rect(&mut canvas, face, bbox_color);
        for (j, &color) in landmark_colors.iter().enumerate() {
            let lx = face.landmarks[j][0] as i32;
            let ly = face.landmarks[j][1] as i32;
            draw_circle(&mut canvas, lx, ly, 3, color);
        }
    }

    let output_path = "images/arnold_arcface.jpg";
    canvas.save(output_path).expect("Failed to save output");
    println!("Saved to {}", output_path);
}

fn draw_rect(img: &mut RgbImage, face: &FaceResult, color: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let x0 = (face.x as i32).clamp(0, w - 1);
    let y0 = (face.y as i32).clamp(0, h - 1);
    let x1 = ((face.x + face.width) as i32).clamp(0, w - 1);
    let y1 = ((face.y + face.height) as i32).clamp(0, h - 1);

    for x in x0..=x1 {
        img.put_pixel(x as u32, y0 as u32, color);
        img.put_pixel(x as u32, y1 as u32, color);
    }
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
