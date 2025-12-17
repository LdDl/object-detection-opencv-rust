//! Ultralytics YOLO models (v8, v9, v11) using ONNX Runtime.

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::inputs;
use ort::value::TensorRef;

#[cfg(feature = "ort-cuda-backend")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "ort-tensorrt-backend")]
use ort::execution_providers::TensorRTExecutionProvider;

use crate::bbox::BBox;
use crate::image_buffer::ImageBuffer;
use crate::postprocess::{Detection, nms, filter_by_class, detections_to_vecs, argmax};
use crate::preprocessing::{preprocess, PreprocessMeta};

/// Error type for ORT model operations.
#[derive(Debug)]
pub enum OrtModelError {
    /// Error from ONNX Runtime
    Ort(ort::Error),
    /// Invalid model output shape
    InvalidOutputShape(String),
    /// Preprocessing error
    PreprocessingError(String),
}

impl std::fmt::Display for OrtModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrtModelError::Ort(e) => write!(f, "ORT error: {}", e),
            OrtModelError::InvalidOutputShape(s) => write!(f, "Invalid output shape: {}", s),
            OrtModelError::PreprocessingError(s) => write!(f, "Preprocessing error: {}", s),
        }
    }
}

impl std::error::Error for OrtModelError {}

impl From<ort::Error> for OrtModelError {
    fn from(e: ort::Error) -> Self {
        OrtModelError::Ort(e)
    }
}

/// Ultralytics YOLO model (v8, v9, v11) using ONNX Runtime.
///
/// This model supports YOLOv8, v9, and v11 which share the same output format.
pub struct ModelUltralyticsOrt {
    session: Session,
    input_width: u32,
    input_height: u32,
    class_filters: Vec<usize>,
    use_letterbox: bool,
}

impl ModelUltralyticsOrt {
    /// Creates a new model from an ONNX file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `class_filters` - List of class indices to detect (empty for all classes)
    ///
    /// # Example
    /// ```ignore
    /// let model = ModelUltralyticsOrt::new_from_file(
    ///     "yolov8n.onnx",
    ///     (640, 640),
    ///     vec![],  // detect all classes
    /// )?;
    /// ```
    pub fn new_from_file(
        model_path: &str,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            input_width: input_size.0,
            input_height: input_size.1,
            class_filters,
            #[cfg(feature = "letterbox")]
            use_letterbox: true,
            #[cfg(not(feature = "letterbox"))]
            use_letterbox: false,
        })
    }

    /// Creates a new model from an ONNX file with CUDA acceleration.
    ///
    /// Requires the `ort-cuda-backend` feature.
    #[cfg(feature = "ort-cuda-backend")]
    pub fn new_from_file_cuda(
        model_path: &str,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            input_width: input_size.0,
            input_height: input_size.1,
            class_filters,
            #[cfg(feature = "letterbox")]
            use_letterbox: true,
            #[cfg(not(feature = "letterbox"))]
            use_letterbox: false,
        })
    }

    /// Creates a new model from an ONNX file with TensorRT acceleration.
    ///
    /// Requires the `ort-tensorrt-backend` feature.
    #[cfg(feature = "ort-tensorrt-backend")]
    pub fn new_from_file_tensorrt(
        model_path: &str,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_execution_providers([TensorRTExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            input_width: input_size.0,
            input_height: input_size.1,
            class_filters,
            #[cfg(feature = "letterbox")]
            use_letterbox: true,
            #[cfg(not(feature = "letterbox"))]
            use_letterbox: false,
        })
    }

    /// Creates a new model with custom session options.
    ///
    /// # Arguments
    /// * `session` - Pre-configured ORT session
    /// * `input_size` - Model input size as (width, height)
    /// * `class_filters` - List of class indices to detect
    pub fn from_session(
        session: Session,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Self {
        Self {
            session,
            input_width: input_size.0,
            input_height: input_size.1,
            class_filters,
            #[cfg(feature = "letterbox")]
            use_letterbox: true,
            #[cfg(not(feature = "letterbox"))]
            use_letterbox: false,
        }
    }

    /// Enables or disables letterbox preprocessing.
    ///
    /// Letterbox preserves aspect ratio by padding with gray.
    /// Default is `false` (stretch mode).
    pub fn set_letterbox(&mut self, enabled: bool) {
        self.use_letterbox = enabled;
    }

    /// Returns the input size (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Runs inference on an image.
    ///
    /// # Arguments
    /// * `image` - Input image buffer
    /// * `conf_threshold` - Confidence threshold (0.0 - 1.0)
    /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
    ///
    /// # Returns
    /// Tuple of (bounding boxes, class IDs, confidence scores)
    pub fn forward(
        &mut self,
        image: &ImageBuffer,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), OrtModelError> {
        // Preprocess
        let (tensor, meta) = preprocess(
            image,
            self.input_width,
            self.input_height,
            self.use_letterbox,
        );

        // Run inference using TensorRef (no copy)
        let outputs = self.session.run(
            inputs!["images" => TensorRef::from_array_view(&tensor)?]
        )?;

        // Get output tensor by name and extract as owned ndarray
        let output = outputs["output0"]
            .try_extract_array::<f32>()?
            .into_owned();  // Make owned copy to avoid borrow conflict

        // Copy class filters to avoid borrow conflict
        let class_filters = self.class_filters.clone();

        // Parse output based on shape
        // YOLOv8/v9/v11 output shape: [1, 84, num_predictions] or [1, num_classes+4, num_predictions]
        let detections = Self::parse_output_array_static(&output.view(), conf_threshold, &meta)?;

        // Apply class filter
        let filtered = filter_by_class(&detections, &class_filters);

        // Apply NMS
        let final_detections = nms(&filtered, nms_threshold);

        Ok(detections_to_vecs(final_detections))
    }

    /// Parses the model output array into detections (static method).
    fn parse_output_array_static(
        output: &ndarray::ArrayViewD<f32>,
        conf_threshold: f32,
        meta: &PreprocessMeta,
    ) -> Result<Vec<Detection>, OrtModelError> {
        let shape = output.shape();

        // Expected shape: [1, 84, num_predictions] for COCO (80 classes + 4 bbox coords)
        // Or more generally: [1, num_classes + 4, num_predictions]
        if shape.len() != 3 || shape[0] != 1 {
            return Err(OrtModelError::InvalidOutputShape(format!(
                "Expected shape [1, C, N], got {:?}",
                shape
            )));
        }

        let num_features = shape[1]; // 84 for COCO
        let num_predictions = shape[2];

        let mut detections = Vec::new();

        // Iterate over predictions
        for i in 0..num_predictions {
            // Extract bbox coords (first 4 values)
            let cx = output[[0, 0, i]];
            let cy = output[[0, 1, i]];
            let w = output[[0, 2, i]];
            let h = output[[0, 3, i]];

            // Extract class scores (remaining values)
            let class_scores: Vec<f32> = (4..num_features)
                .map(|j| output[[0, j, i]])
                .collect();

            // Find best class
            if let Some((class_idx, max_score)) = argmax(&class_scores) {
                if max_score >= conf_threshold {
                    // Transform coordinates back to original image space
                    let (x_orig, y_orig, w_orig, h_orig) = meta.inverse_transform(cx, cy, w, h);

                    let bbox = BBox::from_center(x_orig, y_orig, w_orig, h_orig);

                    detections.push(Detection::new(bbox, class_idx, max_score));
                }
            }
        }

        Ok(detections)
    }
}

impl crate::ObjectDetector for ModelUltralyticsOrt {
    type Input = ImageBuffer;
    type Error = OrtModelError;

    fn detect(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), Self::Error> {
        self.forward(input, conf_threshold, nms_threshold)
    }
}

// OpenCV compatibility: implement ModelTrait for Mat input
#[cfg(feature = "ort-opencv-compat")]
mod opencv_compat_impl {
    use super::*;
    use opencv::core::{Mat, Rect};
    use opencv::Error as OpenCvError;

    impl ModelUltralyticsOrt {
        /// Runs inference on an OpenCV Mat image.
        ///
        /// This is the optimized path that:
        /// 1. Uses OpenCV for resize (works with BGR natively)
        /// 2. Converts BGR→RGB fused with normalization
        /// 3. Runs ORT inference
        ///
        /// # Arguments
        /// * `image` - Input BGR Mat (from VideoCapture, imread, etc.)
        /// * `conf_threshold` - Confidence threshold (0.0 - 1.0)
        /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
        ///
        /// # Returns
        /// Tuple of (bounding boxes as opencv::Rect, class IDs, confidence scores)
        pub fn forward_mat(
            &mut self,
            image: &Mat,
            conf_threshold: f32,
            nms_threshold: f32,
        ) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), OpenCvError> {
            // Use optimized preprocessing: OpenCV resize + fused BGR→RGB conversion
            let (tensor, meta) = crate::opencv_compat::preprocess_mat(
                image,
                self.input_width,
                self.input_height,
                self.use_letterbox,
            )?;

            // Run inference
            let outputs = self.session.run(
                inputs!["images" => TensorRef::from_array_view(&tensor).map_err(|e| {
                    OpenCvError::new(opencv::core::StsError, format!("ORT error: {}", e))
                })?]
            ).map_err(|e| {
                OpenCvError::new(opencv::core::StsError, format!("ORT inference error: {}", e))
            })?;

            // Get output tensor
            let output = outputs["output0"]
                .try_extract_array::<f32>()
                .map_err(|e| {
                    OpenCvError::new(opencv::core::StsError, format!("Output extraction error: {}", e))
                })?
                .into_owned();

            // Parse output
            let detections = Self::parse_output_array_static(&output.view(), conf_threshold, &meta)
                .map_err(|e| {
                    OpenCvError::new(opencv::core::StsError, format!("Parse error: {}", e))
                })?;

            // Apply class filter
            let class_filters = self.class_filters.clone();
            let filtered = filter_by_class(&detections, &class_filters);

            // Apply NMS
            let final_detections = nms(&filtered, nms_threshold);

            // Convert to OpenCV format
            let (bboxes, class_ids, confidences) = detections_to_vecs(final_detections);

            // Convert BBox to opencv::Rect
            let rects: Vec<Rect> = bboxes
                .into_iter()
                .map(|bbox| Rect::new(bbox.x, bbox.y, bbox.width, bbox.height))
                .collect();

            Ok((rects, class_ids, confidences))
        }
    }

    // Implement ModelTrait for ORT model when ort-opencv-compat is enabled
    // Uses the ModelTrait from opencv_compat module (does NOT depend on DNN)
    impl crate::opencv_compat::ModelTrait for ModelUltralyticsOrt {
        fn forward(
            &mut self,
            image: &Mat,
            conf_threshold: f32,
            nms_threshold: f32,
        ) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), OpenCvError> {
            self.forward_mat(image, conf_threshold, nms_threshold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation_error() {
        // Should fail with non-existent file
        let result = ModelUltralyticsOrt::new_from_file(
            "nonexistent.onnx",
            (640, 640),
            vec![],
        );
        assert!(result.is_err());
    }
}
