//! YOLOv5 model using ONNX Runtime.
//!
//! YOLOv5 differs from YOLOv8/v9/v11 in output format:
//! - Output shape: `[1, num_predictions, 85]` (for COCO) vs `[1, 84, num_predictions]`
//! - Has objectness score at index 4
//! - Class scores at indices 5-84
//! - Final confidence = objectness * max_class_score

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
use crate::preprocessing::{preprocess_into, PreprocessMeta};

use super::OrtModelError;

/// YOLOv5 model using ONNX Runtime.
///
/// This model handles the YOLOv5 output format which includes an objectness score.
pub struct ModelYOLOv5Ort {
    session: Session,
    input_width: u32,
    input_height: u32,
    class_filters: Vec<usize>,
    use_letterbox: bool,
    /// Pre-allocated NCHW f32 tensor buffer (avoids per-frame allocation).
    tensor_buf: ndarray::Array4<f32>,
}

impl ModelYOLOv5Ort {
    /// Creates a new YOLOv5 model from an ONNX file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `class_filters` - List of class indices to detect (empty for all classes)
    ///
    /// # Example
    /// ```ignore
    /// let model = ModelYOLOv5Ort::new_from_file(
    ///     "yolov5s.onnx",
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
            tensor_buf: ndarray::Array4::zeros((1, 3, input_size.1 as usize, input_size.0 as usize)),
        })
    }

    /// Creates a new YOLOv5 model from an ONNX file with CUDA acceleration.
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
            tensor_buf: ndarray::Array4::zeros((1, 3, input_size.1 as usize, input_size.0 as usize)),
        })
    }

    /// Creates a new YOLOv5 model from an ONNX file with TensorRT acceleration.
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
            tensor_buf: ndarray::Array4::zeros((1, 3, input_size.1 as usize, input_size.0 as usize)),
        })
    }

    /// Creates a new YOLOv5 model with custom session options.
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
            tensor_buf: ndarray::Array4::zeros((1, 3, input_size.1 as usize, input_size.0 as usize)),
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
        // Preprocess into pre-allocated buffer (zero allocation)
        let meta = preprocess_into(
            image,
            &mut self.tensor_buf,
            self.use_letterbox,
        );

        // Run inference using TensorRef (no copy)
        let outputs = self.session.run(
            inputs!["images" => TensorRef::from_array_view(&self.tensor_buf)?]
        )?;

        // Get output tensor by name and extract as owned ndarray
        let output = outputs["output0"]
            .try_extract_array::<f32>()?
            .into_owned();

        // Copy class filters to avoid borrow conflict
        let class_filters = self.class_filters.clone();

        // Parse output - YOLOv5 format: [1, num_predictions, 85]
        let detections = Self::parse_output_array_static(
            &output.view(),
            conf_threshold,
            &meta,
            self.input_width,
            self.input_height,
        )?;

        // Apply class filter
        let filtered = filter_by_class(&detections, &class_filters);

        // Apply NMS
        let final_detections = nms(&filtered, nms_threshold);

        Ok(detections_to_vecs(final_detections))
    }

    /// Parses the YOLOv5 model output array into detections.
    ///
    /// YOLOv5 output format: `[1, num_predictions, num_features]`
    /// where num_features = 5 + num_classes (85 for COCO)
    /// - indices 0-3: cx, cy, w, h (center coordinates and dimensions)
    /// - index 4: objectness score
    /// - indices 5+: class scores
    fn parse_output_array_static(
        output: &ndarray::ArrayViewD<f32>,
        conf_threshold: f32,
        meta: &PreprocessMeta,
        input_width: u32,
        input_height: u32,
    ) -> Result<Vec<Detection>, OrtModelError> {
        let shape = output.shape();

        // Expected shape: [1, num_predictions, 85] for COCO (4 bbox + 1 obj + 80 classes)
        if shape.len() != 3 || shape[0] != 1 {
            return Err(OrtModelError::InvalidOutputShape(format!(
                "Expected shape [1, N, C], got {:?}",
                shape
            )));
        }

        let num_predictions = shape[1];
        let num_features = shape[2]; // 85 for COCO

        if num_features < 6 {
            return Err(OrtModelError::InvalidOutputShape(format!(
                "Expected at least 6 features (4 bbox + 1 obj + 1 class), got {}",
                num_features
            )));
        }

        let mut detections = Vec::new();

        // Iterate over predictions
        for i in 0..num_predictions {
            // Extract objectness score at index 4
            let objectness = output[[0, i, 4]];

            // Early filter by objectness threshold
            if objectness < conf_threshold {
                continue;
            }

            // Extract class scores (indices 5 to num_features)
            let class_scores: Vec<f32> = (5..num_features)
                .map(|j| output[[0, i, j]])
                .collect();

            // Find best class
            if let Some((class_idx, max_class_score)) = argmax(&class_scores) {
                // YOLOv5 confidence = objectness * class_score
                let confidence = objectness * max_class_score;

                if confidence >= conf_threshold {
                    // Extract bbox coords (first 4 values)
                    let mut cx = output[[0, i, 0]];
                    let mut cy = output[[0, i, 1]];
                    let mut w = output[[0, i, 2]];
                    let mut h = output[[0, i, 3]];

                    // Handle normalized coordinates (if all values < 2.0, assume normalized)
                    // YOLOv5 exports can have either pixel or normalized coordinates
                    if cx < 2.0 && cy < 2.0 && w < 2.0 && h < 2.0 {
                        cx *= input_width as f32;
                        cy *= input_height as f32;
                        w *= input_width as f32;
                        h *= input_height as f32;
                    }

                    // Transform coordinates back to original image space
                    let (x_orig, y_orig, w_orig, h_orig) = meta.inverse_transform(cx, cy, w, h);

                    let bbox = BBox::from_center(x_orig, y_orig, w_orig, h_orig);

                    detections.push(Detection::new(bbox, class_idx, confidence));
                }
            }
        }

        Ok(detections)
    }
}

impl crate::ObjectDetector for ModelYOLOv5Ort {
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

    impl ModelYOLOv5Ort {
        /// Runs inference on an OpenCV Mat image.
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
            // Use optimized preprocessing: OpenCV resize + fused BGR->RGB conversion
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
            let detections = Self::parse_output_array_static(
                &output.view(),
                conf_threshold,
                &meta,
                self.input_width,
                self.input_height,
            ).map_err(|e| {
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

    // Implement ModelTrait for YOLOv5 model when ort-opencv-compat is enabled
    impl crate::opencv_compat::ModelTrait for ModelYOLOv5Ort {
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
        let result = ModelYOLOv5Ort::new_from_file(
            "nonexistent.onnx",
            (640, 640),
            vec![],
        );
        assert!(result.is_err());
    }
}
