//! YuNet face detection model using OpenCV's FaceDetectorYN.
//!
//! Model: face_detection_yunet_2023mar (OpenCV Zoo)
//! Uses OpenCV's built-in FaceDetectorYN (objdetect module, OpenCV 4.5.4+).
//! All preprocessing, inference, decoding and NMS are handled by OpenCV internally.

use opencv::core::{Mat, Ptr, Size};
use opencv::objdetect::FaceDetectorYN;
use opencv::prelude::*;

use crate::face_detection::{FaceDetection, FaceDetector};

/// YuNet face detection model using OpenCV's FaceDetectorYN.
///
/// This is a thin wrapper around OpenCV's built-in face detector.
/// Requires OpenCV 4.5.4+ with the `objdetect` module.
pub struct ModelYuNetOpenCV {
    detector: Ptr<FaceDetectorYN>,
    input_width: u32,
    input_height: u32,
}

impl ModelYuNetOpenCV {
    /// Creates a new YuNet model from an ONNX file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the YuNet ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `backend` - DNN backend (e.g., `DNN_BACKEND_DEFAULT`, `DNN_BACKEND_CUDA`)
    /// * `target` - DNN target device (e.g., `DNN_TARGET_CPU`, `DNN_TARGET_CUDA`)
    pub fn new_from_file(
        model_path: &str,
        input_size: (i32, i32),
        backend: i32,
        target: i32,
    ) -> Result<Self, opencv::Error> {
        let detector = FaceDetectorYN::create(
            model_path,
            "",
            Size::new(input_size.0, input_size.1),
            0.9,    // score_threshold (will be overridden per-call)
            0.3,    // nms_threshold (will be overridden per-call)
            5000,   // top_k
            backend,
            target,
        )?;

        Ok(Self {
            detector,
            input_width: input_size.0 as u32,
            input_height: input_size.1 as u32,
        })
    }

    /// Returns the input size (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Runs face detection on an OpenCV Mat image.
    ///
    /// # Arguments
    /// * `image` - Input BGR Mat image
    /// * `conf_threshold` - Confidence threshold (0.0 - 1.0)
    /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
    pub fn forward(
        &mut self,
        image: &Mat,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, opencv::Error> {
        // Update thresholds
        self.detector.set_score_threshold(conf_threshold)?;
        self.detector.set_nms_threshold(nms_threshold)?;

        // Set input size to match the actual image
        let img_size = image.size()?;
        self.detector.set_input_size(img_size)?;

        // Detect
        let mut faces = Mat::default();
        self.detector.detect(image, &mut faces)?;

        // Parse Nx15 output
        let num_faces = faces.rows();
        let mut detections = Vec::with_capacity(num_faces as usize);

        for i in 0..num_faces {
            let x = *faces.at_2d::<f32>(i, 0)?;
            let y = *faces.at_2d::<f32>(i, 1)?;
            let w = *faces.at_2d::<f32>(i, 2)?;
            let h = *faces.at_2d::<f32>(i, 3)?;

            let mut landmarks = [[0.0f32; 2]; 5];
            for k in 0..5 {
                landmarks[k][0] = *faces.at_2d::<f32>(i, 4 + k as i32 * 2)?;
                landmarks[k][1] = *faces.at_2d::<f32>(i, 5 + k as i32 * 2)?;
            }

            let confidence = *faces.at_2d::<f32>(i, 14)?;

            detections.push(FaceDetection {
                x,
                y,
                width: w,
                height: h,
                landmarks,
                confidence,
            });
        }

        Ok(detections)
    }
}

impl FaceDetector for ModelYuNetOpenCV {
    type Input = Mat;
    type Error = opencv::Error;

    fn detect_faces(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, Self::Error> {
        self.forward(input, conf_threshold, nms_threshold)
    }
}
