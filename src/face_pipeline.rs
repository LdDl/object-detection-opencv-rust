//! Face detection + recognition pipeline.
//!
//! Combines YuNet face detection, landmark-based alignment, and ArcFace
//! embedding extraction into a single pipeline.
//!
//! # Pipeline
//!
//! ```text
//! Image -> YuNet (detect) -> 5 landmarks -> Affine warp (112x112) -> ArcFace (embed) -> [f32; 512]
//! ```

use crate::backend_ort::{ModelYuNetOrt, ModelArcFaceOrt, ArcFaceNorm, OrtModelError};
use crate::face_alignment::align_face;
use crate::face_detection::FaceDetection;
use crate::image_buffer::ImageBuffer;

/// Result of face detection + recognition for a single face.
#[derive(Debug, Clone)]
pub struct FaceResult {
    /// Bounding box top-left X
    pub x: f32,
    /// Bounding box top-left Y
    pub y: f32,
    /// Bounding box width
    pub width: f32,
    /// Bounding box height
    pub height: f32,
    /// Detection confidence
    pub confidence: f32,
    /// 5 facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
    pub landmarks: [[f32; 2]; 5],
    /// 512-dimensional L2-normalized embedding
    pub embedding: [f32; 512],
}

/// Face detection + recognition pipeline using ORT backend.
///
/// Combines YuNet detector and ArcFace recognizer into a unified pipeline.
///
/// # Example
/// ```ignore
/// let mut pipeline = FacePipeline::new(
///     "face_detection_yunet_2023mar.onnx",
///     "w600k_mbf.onnx",
/// )?;
///
/// let faces = pipeline.process(&image, 0.7, 0.3)?;
/// for face in &faces {
///     println!("confidence={:.3}, embedding_norm={:.3}",
///         face.confidence,
///         face.embedding.iter().map(|v| v * v).sum::<f32>().sqrt());
/// }
/// ```
pub struct FacePipeline {
    detector: ModelYuNetOrt,
    recognizer: ModelArcFaceOrt,
}

impl FacePipeline {
    /// Creates a new face pipeline with CPU inference.
    ///
    /// # Arguments
    /// * `detector_path` - Path to YuNet ONNX model
    /// * `recognizer_path` - Path to ArcFace ONNX model (e.g. `w600k_mbf.onnx`)
    pub fn new(
        detector_path: &str,
        recognizer_path: &str,
    ) -> Result<Self, OrtModelError> {
        Self::new_with_norm(detector_path, recognizer_path, ArcFaceNorm::default())
    }

    /// Creates a new face pipeline with CPU inference and explicit normalization.
    ///
    /// # Example
    /// ```ignore
    /// // Use ResNet50 model
    /// let pipeline = FacePipeline::new_with_norm(
    ///     "yunet.onnx", "w600k_r50.onnx", ArcFaceNorm::ResNet,
    /// )?;
    /// ```
    pub fn new_with_norm(
        detector_path: &str,
        recognizer_path: &str,
        norm: ArcFaceNorm,
    ) -> Result<Self, OrtModelError> {
        let detector = ModelYuNetOrt::new_from_file(detector_path)?;
        let recognizer = ModelArcFaceOrt::new_from_file_with_norm(recognizer_path, norm)?;
        Ok(Self { detector, recognizer })
    }

    /// Creates a new face pipeline with CUDA inference.
    #[cfg(feature = "ort-cuda-backend")]
    pub fn new_cuda(
        detector_path: &str,
        recognizer_path: &str,
    ) -> Result<Self, OrtModelError> {
        Self::new_cuda_with_norm(detector_path, recognizer_path, ArcFaceNorm::default())
    }

    /// Creates a new face pipeline with CUDA inference and explicit normalization.
    #[cfg(feature = "ort-cuda-backend")]
    pub fn new_cuda_with_norm(
        detector_path: &str,
        recognizer_path: &str,
        norm: ArcFaceNorm,
    ) -> Result<Self, OrtModelError> {
        let detector = ModelYuNetOrt::new_from_file_cuda(detector_path)?;
        let recognizer = ModelArcFaceOrt::new_from_file_cuda_with_norm(recognizer_path, norm)?;
        Ok(Self { detector, recognizer })
    }

    /// Creates a new face pipeline with TensorRT acceleration via ORT.
    #[cfg(feature = "ort-tensorrt-backend")]
    pub fn new_tensorrt(
        detector_path: &str,
        recognizer_path: &str,
    ) -> Result<Self, OrtModelError> {
        Self::new_tensorrt_with_norm(detector_path, recognizer_path, ArcFaceNorm::default())
    }

    /// Creates a new face pipeline with TensorRT acceleration and explicit normalization.
    #[cfg(feature = "ort-tensorrt-backend")]
    pub fn new_tensorrt_with_norm(
        detector_path: &str,
        recognizer_path: &str,
        norm: ArcFaceNorm,
    ) -> Result<Self, OrtModelError> {
        let detector = ModelYuNetOrt::new_from_file_tensorrt(detector_path)?;
        let recognizer = ModelArcFaceOrt::new_from_file_tensorrt_with_norm(recognizer_path, norm)?;
        Ok(Self { detector, recognizer })
    }

    /// Returns the detector's input size (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        self.detector.input_size()
    }

    /// Enables or disables letterbox preprocessing for the detector.
    pub fn set_letterbox(&mut self, enabled: bool) {
        self.detector.set_letterbox(enabled);
    }

    /// Detects all faces and extracts embeddings.
    ///
    /// # Arguments
    /// * `image` - Input image (RGB, any size)
    /// * `conf_threshold` - Detection confidence threshold (0.0 - 1.0)
    /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
    ///
    /// # Returns
    /// A vector of `FaceResult` with detection info and embeddings.
    pub fn process(
        &mut self,
        image: &ImageBuffer,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceResult>, OrtModelError> {
        let detections = self.detector.forward(image, conf_threshold, nms_threshold)?;

        let mut results = Vec::with_capacity(detections.len());
        for det in &detections {
            let aligned = align_face(image, &det.landmarks);
            let embedding = self.recognizer.forward(&aligned)?;
            results.push(FaceResult {
                x: det.x,
                y: det.y,
                width: det.width,
                height: det.height,
                confidence: det.confidence,
                landmarks: det.landmarks,
                embedding,
            });
        }

        Ok(results)
    }

    /// Extracts embedding from a pre-aligned 112x112 face.
    ///
    /// Use this when alignment is done externally.
    pub fn embed(&mut self, aligned_face: &ImageBuffer) -> Result<[f32; 512], OrtModelError> {
        self.recognizer.forward(aligned_face)
    }

    /// Runs detection only (without recognition).
    ///
    /// Useful for getting bounding boxes and landmarks without embedding extraction.
    pub fn detect(
        &mut self,
        image: &ImageBuffer,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, OrtModelError> {
        self.detector.forward(image, conf_threshold, nms_threshold)
    }
}

/// Computes cosine similarity between two L2-normalized embeddings.
///
/// For L2-normalized vectors, cosine similarity equals the dot product.
///
/// # Returns
/// Similarity score in [-1.0, 1.0]. Higher means more similar.
pub fn cosine_similarity(a: &[f32; 512], b: &[f32; 512]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..512 {
        dot += a[i] * b[i];
    }
    dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let mut a = [0.0f32; 512];
        // Create a unit vector
        a[0] = 1.0;
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = [0.0f32; 512];
        let mut b = [0.0f32; 512];
        a[0] = 1.0;
        b[1] = 1.0;
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let mut a = [0.0f32; 512];
        let mut b = [0.0f32; 512];
        a[0] = 1.0;
        b[0] = -1.0;
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    /// Integration test: full pipeline on arnold.jpg.
    ///
    /// Requires model weights in pretrained/:
    /// - face_detection_yunet_2023mar.onnx
    /// - w600k_mbf.onnx
    ///
    /// Run with: cargo test --features ort-backend -- --ignored test_pipeline_arnold
    #[test]
    #[ignore]
    fn test_pipeline_arnold() {
        ort::init().commit();

        let detector_path = "pretrained/face_detection_yunet_2023mar.onnx";
        let recognizer_path = "pretrained/w600k_mbf.onnx";
        let image_path = "images/arnold.jpg";

        // Skip if models or image are missing
        if !std::path::Path::new(detector_path).exists()
            || !std::path::Path::new(recognizer_path).exists()
            || !std::path::Path::new(image_path).exists()
        {
            eprintln!("Skipping test_pipeline_arnold: model or image files not found");
            return;
        }

        let mut pipeline = FacePipeline::new(detector_path, recognizer_path)
            .expect("Failed to create pipeline");

        let img = image::open(image_path).expect("Failed to load image");
        let img_buffer = ImageBuffer::from_dynamic_image(img);

        let faces = pipeline.process(&img_buffer, 0.7, 0.3)
            .expect("Pipeline failed");

        // Arnold.jpg should have at least one face
        assert!(!faces.is_empty(), "No faces detected in arnold.jpg");

        for (i, face) in faces.iter().enumerate() {
            // Confidence should be reasonable
            assert!(face.confidence > 0.5, "Face #{} confidence too low: {}", i, face.confidence);

            // Embedding should be L2-normalized (norm ≈ 1.0)
            let norm: f32 = face.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Face #{} embedding L2 norm is {}, expected ~1.0", i, norm
            );

            // Embedding should not be all zeros
            let nonzero = face.embedding.iter().any(|&v| v.abs() > 1e-6);
            assert!(nonzero, "Face #{} embedding is all zeros", i);
        }
    }

    /// Test alignment produces correct output for a real image.
    #[test]
    #[ignore]
    fn test_alignment_on_arnold() {
        ort::init().commit();

        let detector_path = "pretrained/face_detection_yunet_2023mar.onnx";
        let image_path = "images/arnold.jpg";

        if !std::path::Path::new(detector_path).exists()
            || !std::path::Path::new(image_path).exists()
        {
            eprintln!("Skipping test_alignment_on_arnold: files not found");
            return;
        }

        let mut detector = crate::backend_ort::ModelYuNetOrt::new_from_file(detector_path)
            .expect("Failed to load YuNet");

        let img = image::open(image_path).expect("Failed to load image");
        let img_buffer = ImageBuffer::from_dynamic_image(img);

        let detections = detector.forward(&img_buffer, 0.7, 0.3)
            .expect("Detection failed");

        assert!(!detections.is_empty(), "No faces detected");

        let aligned = align_face(&img_buffer, &detections[0].landmarks);
        assert_eq!(aligned.width(), 112);
        assert_eq!(aligned.height(), 112);
        assert_eq!(aligned.channels(), 3);

        // Check that aligned face is not all black (warp produced actual content)
        let data = aligned.as_array();
        let sum: u64 = data.iter().map(|&v| v as u64).sum();
        assert!(sum > 0, "Aligned face is all black");
    }

    /// Compare MobileFaceNet and ResNet50 embeddings on the same image.
    ///
    /// Both models should detect the same face and produce similar (but not identical)
    /// embeddings due to different architectures.
    ///
    /// Run with: cargo test --lib --features ort-backend test_mbf_vs_r50 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_mbf_vs_r50() {
        ort::init().commit();

        let detector_path = "pretrained/face_detection_yunet_2023mar.onnx";
        let mbf_path = "pretrained/w600k_mbf.onnx";
        let r50_path = "pretrained/w600k_r50.onnx";
        let image_path = "images/arnold.jpg";

        if !std::path::Path::new(detector_path).exists()
            || !std::path::Path::new(mbf_path).exists()
            || !std::path::Path::new(r50_path).exists()
            || !std::path::Path::new(image_path).exists()
        {
            eprintln!("Skipping test_mbf_vs_r50: model or image files not found");
            return;
        }

        let mut pipeline_mbf = FacePipeline::new(detector_path, mbf_path)
            .expect("Failed to create MBF pipeline");
        let mut pipeline_r50 = FacePipeline::new_with_norm(
            detector_path, r50_path, ArcFaceNorm::ResNet,
        ).expect("Failed to create R50 pipeline");

        let img = image::open(image_path).expect("Failed to load image");
        let img_buffer = ImageBuffer::from_dynamic_image(img);

        let faces_mbf = pipeline_mbf.process(&img_buffer, 0.7, 0.3)
            .expect("MBF pipeline failed");
        let faces_r50 = pipeline_r50.process(&img_buffer, 0.7, 0.3)
            .expect("R50 pipeline failed");

        assert!(!faces_mbf.is_empty(), "MBF: no faces detected");
        assert!(!faces_r50.is_empty(), "R50: no faces detected");

        let norm_mbf: f32 = faces_mbf[0].embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_r50: f32 = faces_r50[0].embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("MBF L2 norm: {:.4}", norm_mbf);
        println!("R50 L2 norm: {:.4}", norm_r50);

        assert!((norm_mbf - 1.0).abs() < 0.01, "MBF norm not ~1.0: {}", norm_mbf);
        assert!((norm_r50 - 1.0).abs() < 0.01, "R50 norm not ~1.0: {}", norm_r50);

        // Cross-model similarity: informational only.
        // Different architectures produce incompatible embedding spaces,
        // so this value is NOT meaningful for identity comparison.
        let cross_sim = cosine_similarity(&faces_mbf[0].embedding, &faces_r50[0].embedding);
        println!("MBF vs R50 cosine similarity (informational): {:.4}", cross_sim);
    }
}
