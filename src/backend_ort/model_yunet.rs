//! YuNet face detection model using ONNX Runtime.
//!
//! Model: face_detection_yunet_2023mar (OpenCV Zoo)
//! Input: [1, 3, H, W] float32, BGR, [0..255]
//! Outputs: 12 tensors (4 per stride 8/16/32): cls, obj, bbox, kps

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::inputs;
use ort::value::TensorRef;

use ndarray::Array4;

use crate::face_detection::{FaceDetection, FaceDetector, STRIDES, decode_yunet_stride, nms_faces};
use crate::image_buffer::ImageBuffer;
use crate::preprocessing::{PreprocessMeta, preprocess_into_bgr, preprocess_into_bgr_letterbox};

use super::OrtModelError;

/// YuNet face detection model using ONNX Runtime.
///
/// Extremely lightweight (0.083M params, 228K ONNX file).
/// Auto-detects input dimensions from ONNX metadata.
/// Uses letterbox preprocessing by default to preserve aspect ratio.
pub struct ModelYuNetOrt {
    session: Session,
    input_width: u32,
    input_height: u32,
    tensor_buf: Array4<f32>,
    use_letterbox: bool,
}

impl ModelYuNetOrt {
    /// Creates a new YuNet model from an ONNX file (CPU).
    ///
    /// Input dimensions are read from the model metadata automatically.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Example
    /// ```ignore
    /// let model = ModelYuNetOrt::new_from_file("face_detection_yunet_2023mar.onnx")?;
    /// ```
    pub fn new_from_file(model_path: &str) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Self::from_session(session)
    }

    /// Creates a new YuNet model with CUDA acceleration.
    #[cfg(feature = "ort-cuda-backend")]
    pub fn new_from_file_cuda(model_path: &str) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Self::from_session(session)
    }

    /// Creates a new YuNet model with TensorRT acceleration via ORT.
    #[cfg(feature = "ort-tensorrt-backend")]
    pub fn new_from_file_tensorrt(model_path: &str) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_execution_providers([ort::execution_providers::TensorRTExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Self::from_session(session)
    }

    fn from_session(session: Session) -> Result<Self, OrtModelError> {
        let inputs = session.inputs();
        if inputs.is_empty() {
            return Err(OrtModelError::InvalidOutputShape("YuNet model has no inputs".into()));
        }
        let input_shape = inputs[0].dtype().tensor_shape()
            .ok_or_else(|| OrtModelError::InvalidOutputShape("Cannot read input tensor shape".into()))?;
        if input_shape.len() != 4 {
            return Err(OrtModelError::InvalidOutputShape(
                format!("Expected 4D input, got {}D", input_shape.len()),
            ));
        }
        let input_height = input_shape[2] as u32;
        let input_width = input_shape[3] as u32;

        let tensor_buf = Array4::<f32>::zeros((1, 3, input_height as usize, input_width as usize));

        Ok(Self { session, input_width, input_height, tensor_buf, use_letterbox: true })
    }

    /// Enables or disables letterbox preprocessing.
    ///
    /// When enabled (default), preserves aspect ratio and pads with gray.
    /// When disabled, stretches to model input size (may distort proportions).
    pub fn set_letterbox(&mut self, enabled: bool) {
        self.use_letterbox = enabled;
    }

    /// Returns the input size (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Runs face detection on an image.
    ///
    /// # Arguments
    /// * `image` - Input image buffer (RGB)
    /// * `conf_threshold` - Confidence threshold (0.0 - 1.0)
    /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
    ///
    /// # Returns
    /// Vector of face detections
    pub fn forward(
        &mut self,
        image: &ImageBuffer,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, OrtModelError> {
        let meta: PreprocessMeta = if self.use_letterbox {
            PreprocessMeta::Letterbox(preprocess_into_bgr_letterbox(image, &mut self.tensor_buf))
        } else {
            PreprocessMeta::Stretch(preprocess_into_bgr(image, &mut self.tensor_buf))
        };

        let outputs = self.session.run(
            inputs!["input" => TensorRef::from_array_view(&self.tensor_buf)?]
        )?;

        let iw = self.input_width as f32;
        let ih = self.input_height as f32;

        let mut detections = Vec::new();

        for &stride in &STRIDES {
            let feat_w = (iw / stride as f32).ceil() as usize;
            let feat_h = (ih / stride as f32).ceil() as usize;

            let cls_name = format!("cls_{}", stride);
            let obj_name = format!("obj_{}", stride);
            let bbox_name = format!("bbox_{}", stride);
            let kps_name = format!("kps_{}", stride);

            let cls = match outputs[cls_name.as_str()].try_extract_array::<f32>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let obj = match outputs[obj_name.as_str()].try_extract_array::<f32>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let bbox = match outputs[bbox_name.as_str()].try_extract_array::<f32>() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let kps = match outputs[kps_name.as_str()].try_extract_array::<f32>() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let cls_flat = cls.as_standard_layout();
            let obj_flat = obj.as_standard_layout();
            let bbox_flat = bbox.as_standard_layout();
            let kps_flat = kps.as_standard_layout();

            let cls_slice = cls_flat.as_slice().unwrap_or(&[]);
            let obj_slice = obj_flat.as_slice().unwrap_or(&[]);
            let bbox_slice = bbox_flat.as_slice().unwrap_or(&[]);
            let kps_slice = kps_flat.as_slice().unwrap_or(&[]);

            decode_yunet_stride(
                cls_slice, obj_slice, bbox_slice, kps_slice,
                stride, feat_w, feat_h,
                &meta,
                conf_threshold,
                &mut detections,
            );
        }

        Ok(nms_faces(&detections, nms_threshold))
    }
}

impl FaceDetector for ModelYuNetOrt {
    type Input = ImageBuffer;
    type Error = OrtModelError;

    fn detect_faces(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, Self::Error> {
        self.forward(input, conf_threshold, nms_threshold)
    }
}
