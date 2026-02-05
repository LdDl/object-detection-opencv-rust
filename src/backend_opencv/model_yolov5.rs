//! YOLOv5 model using OpenCV DNN.
//!
//! YOLOv5 differs from YOLOv8/v9/v11 in output format:
//! - Output shape: `[1, num_predictions, 85]` (for COCO) vs `[1, 84, num_predictions]`
//! - Has objectness score at index 4
//! - Class scores at indices 5-84
//! - Final confidence = objectness * max_class_score

use opencv::{
    prelude::NetTrait,
    prelude::NetTraitConst,
    prelude::MatTraitConst,
    core::VectorToVec,
    core::Scalar,
    core::Size,
    core::Mat,
    core::Vector,
    core::Rect,
    core::CV_32F,
    dnn::read_net_from_onnx,
    dnn::blob_from_image,
    dnn::nms_boxes,
    dnn::Net,
    Error
};

use crate::BBox;

#[cfg(feature = "letterbox")]
use opencv::{
    core::CV_8UC3,
    core::BORDER_CONSTANT,
    core::copy_make_border,
    imgproc::resize,
    imgproc::INTER_LINEAR,
};

use super::model::ModelTrait;
use super::utils::BACKEND_TARGET_VALID;

const YOLO_BLOB_MEAN: (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);

/// YOLOv5 model using OpenCV DNN.
///
/// This model handles the YOLOv5 output format which includes an objectness score.
/// See the ref. https://github.com/ultralytics/yolov5
pub struct ModelYOLOv5OpenCV {
    // Underlying OpenCV's DNN Net implementation
    net: Net,
    // Input size (width and height)
    input_size: Size,
    // Blob's mean for OpenCV's blob
    blob_mean: Scalar,
    // Blob's scale for OpenCV's blob (1/255 for YOLO)
    blob_scale: f64,
    // Blob's name
    blob_name: &'static str,
    // Output layer names
    out_layers: Vector<String>,
    // Classes to filter (empty = all classes)
    filter_classes: Vec<usize>,
    // Reusable buffer for letterbox resize
    #[cfg(feature = "letterbox")]
    letterbox_resized: Mat,
    // Reusable buffer for letterbox padding
    #[cfg(feature = "letterbox")]
    letterbox_padded: Mat,
}

impl ModelYOLOv5OpenCV {
    /// Creates a new YOLOv5 model from an ONNX file.
    ///
    /// # Arguments
    /// * `weight_file_path` - Path to the ONNX model file
    /// * `net_size` - Model input size as (width, height)
    /// * `backend_id` - OpenCV DNN backend ID
    /// * `target_id` - OpenCV DNN target ID
    /// * `filter_classes` - List of class indices to detect (empty for all)
    ///
    /// # Example
    /// ```ignore
    /// use opencv::dnn::{DNN_BACKEND_CUDA, DNN_TARGET_CUDA};
    /// use od_opencv::backend_opencv::model_yolov5::ModelYOLOv5OpenCV;
    ///
    /// let mut model = ModelYOLOv5OpenCV::new_from_onnx_file(
    ///     "yolov5s.onnx",
    ///     (640, 640),
    ///     DNN_BACKEND_CUDA,
    ///     DNN_TARGET_CUDA,
    ///     vec![],
    /// )?;
    /// ```
    pub fn new_from_onnx_file(
        weight_file_path: &str,
        net_size: (i32, i32),
        backend_id: i32,
        target_id: i32,
        filter_classes: Vec<usize>,
    ) -> Result<Self, Error> {
        if BACKEND_TARGET_VALID.get(&backend_id).and_then(|map| map.get(&target_id)).is_none() {
            return Err(Error::new(400, format!(
                "Combination of BACKEND '{}' and TARGET '{}' is not valid",
                backend_id, target_id
            )));
        };

        let neural_net = read_net_from_onnx(weight_file_path)?;
        Self::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)
    }

    /// Creates a model from an existing OpenCV DNN network.
    pub fn new_from_dnn(
        mut neural_net: Net,
        net_size: (i32, i32),
        backend_id: i32,
        target_id: i32,
        filter_classes: Vec<usize>,
    ) -> Result<Self, Error> {
        neural_net.set_preferable_backend(backend_id)?;
        neural_net.set_preferable_target(target_id)?;
        let out_layers = neural_net.get_unconnected_out_layers_names()?;

        #[cfg(feature = "letterbox")]
        let letterbox_padded = Mat::new_rows_cols_with_default(
            net_size.1,  // height
            net_size.0,  // width
            CV_8UC3,
            Scalar::new(114.0, 114.0, 114.0, 0.0)
        )?;

        Ok(Self {
            net: neural_net,
            input_size: Size::new(net_size.0, net_size.1),
            blob_mean: Scalar::new(YOLO_BLOB_MEAN.0, YOLO_BLOB_MEAN.1, YOLO_BLOB_MEAN.2, YOLO_BLOB_MEAN.3),
            blob_scale: 1.0 / 255.0,
            blob_name: "",
            out_layers,
            filter_classes,
            #[cfg(feature = "letterbox")]
            letterbox_resized: Mat::default(),
            #[cfg(feature = "letterbox")]
            letterbox_padded,
        })
    }

    /// Runs inference on an image.
    ///
    /// # Arguments
    /// * `image` - Input image (BGR format from OpenCV)
    /// * `conf_threshold` - Confidence threshold (0.0 - 1.0)
    /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
    ///
    /// # Returns
    /// Tuple of (bounding boxes, class IDs, confidence scores)
    pub fn forward(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), Error> {
        let image_width = image.cols();
        let image_height = image.rows();

        // Preprocessing and coordinate conversion factors depend on feature flag
        #[cfg(feature = "letterbox")]
        let (blobimg, scale, pad_left, pad_top) = {
            let scale = f32::min(
                self.input_size.width as f32 / image_width as f32,
                self.input_size.height as f32 / image_height as f32
            );
            let new_width = (image_width as f32 * scale).round() as i32;
            let new_height = (image_height as f32 * scale).round() as i32;
            let pad_left = (self.input_size.width - new_width) / 2;
            let pad_top = (self.input_size.height - new_height) / 2;

            let blob = if image_width != self.input_size.width || image_height != self.input_size.height {
                resize(&image, &mut self.letterbox_resized, Size::new(new_width, new_height), 0.0, 0.0, INTER_LINEAR)?;
                let pad_right = self.input_size.width - new_width - pad_left;
                let pad_bottom = self.input_size.height - new_height - pad_top;
                copy_make_border(
                    &self.letterbox_resized,
                    &mut self.letterbox_padded,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    BORDER_CONSTANT,
                    Scalar::new(114.0, 114.0, 114.0, 0.0)
                )?;
                blob_from_image(&self.letterbox_padded, self.blob_scale, Size::new(0, 0), self.blob_mean, true, false, CV_32F)?
            } else {
                blob_from_image(&image, self.blob_scale, self.input_size, self.blob_mean, true, false, CV_32F)?
            };
            (blob, scale, pad_left, pad_top)
        };

        #[cfg(not(feature = "letterbox"))]
        let (blobimg, scale_x, scale_y) = {
            let scale_x = image_width as f32 / self.input_size.width as f32;
            let scale_y = image_height as f32 / self.input_size.height as f32;
            let blob = blob_from_image(&image, self.blob_scale, self.input_size, self.blob_mean, true, false, CV_32F)?;
            (blob, scale_x, scale_y)
        };

        let mut detections = Vector::<Mat>::new();
        self.net.set_input(&blobimg, self.blob_name, 1.0, self.blob_mean)?;
        self.net.forward(&mut detections, &self.out_layers)?;

        // Collect output data
        let mut bboxes = Vector::<Rect>::new();
        let mut confidences = Vector::<f32>::new();
        let mut class_ids = Vec::new();

        // YOLOv5 output format: [1, num_predictions, 85]
        // where 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        for layer in detections {
            let mat_size = layer.mat_size();

            // YOLOv5 outputs [1, num_predictions, num_features]
            // mat_size[0] = 1 (batch)
            // mat_size[1] = num_predictions (e.g., 25200 for 640x640)
            // mat_size[2] = num_features (85 for COCO)
            let num_predictions = mat_size[1];
            let num_features = mat_size[2];

            for i in 0..num_predictions {
                // Access elements: [batch=0, prediction=i, feature=j]
                let objectness = *layer.at_3d::<f32>(0, i, 4)?;

                // Early filter by objectness
                if objectness < conf_threshold {
                    continue;
                }

                // Find max class score
                let mut max_class_index = 0usize;
                let mut max_class_score = 0.0f32;
                for j in 5..num_features {
                    let score = *layer.at_3d::<f32>(0, i, j)?;
                    if score > max_class_score {
                        max_class_score = score;
                        max_class_index = (j - 5) as usize;
                    }
                }

                // YOLOv5 confidence = objectness * class_score
                let confidence = objectness * max_class_score;

                if confidence >= conf_threshold {
                    if !self.filter_classes.is_empty() && !self.filter_classes.contains(&max_class_index) {
                        continue;
                    }

                    // Extract bbox coordinates
                    let mut cx = *layer.at_3d::<f32>(0, i, 0)?;
                    let mut cy = *layer.at_3d::<f32>(0, i, 1)?;
                    let mut w = *layer.at_3d::<f32>(0, i, 2)?;
                    let mut h = *layer.at_3d::<f32>(0, i, 3)?;

                    // Handle normalized coordinates (if values < 2.0, assume normalized)
                    if cx < 2.0 && cy < 2.0 && w < 2.0 && h < 2.0 {
                        cx *= self.input_size.width as f32;
                        cy *= self.input_size.height as f32;
                        w *= self.input_size.width as f32;
                        h *= self.input_size.height as f32;
                    }

                    // Coordinate conversion back to original image space
                    #[cfg(feature = "letterbox")]
                    let (x_center, y_center, width, height) = {
                        (
                            (cx - pad_left as f32) / scale,
                            (cy - pad_top as f32) / scale,
                            w / scale,
                            h / scale,
                        )
                    };

                    #[cfg(not(feature = "letterbox"))]
                    let (x_center, y_center, width, height) = {
                        (
                            cx * scale_x,
                            cy * scale_y,
                            w * scale_x,
                            h * scale_y,
                        )
                    };

                    // Convert from center to top-left corner
                    let bbox_cv = Rect::new(
                        (x_center - width / 2.0).round() as i32,
                        (y_center - height / 2.0).round() as i32,
                        width.round() as i32,
                        height.round() as i32
                    );
                    bboxes.push(bbox_cv);
                    confidences.push(confidence);
                    class_ids.push(max_class_index);
                }
            }
        }

        // Run NMS
        let mut indices = Vector::<i32>::new();
        nms_boxes(&bboxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1.0, 0)?;

        let mut nms_bboxes = vec![];
        let mut nms_classes_ids = vec![];
        let mut nms_confidences = vec![];

        let indices_vec = indices.to_vec();
        let mut bboxes = bboxes.to_vec();
        nms_bboxes.extend(bboxes.drain(..)
            .enumerate()
            .filter_map(|(idx, item)| if indices_vec.contains(&(idx as i32)) { Some(item) } else { None }));

        nms_classes_ids.extend(class_ids.drain(..)
            .enumerate()
            .filter_map(|(idx, item)| if indices_vec.contains(&(idx as i32)) { Some(item) } else { None }));

        nms_confidences.extend(confidences.to_vec().drain(..)
            .enumerate()
            .filter_map(|(idx, item)| if indices_vec.contains(&(idx as i32)) { Some(item) } else { None }));

        Ok((nms_bboxes, nms_classes_ids, nms_confidences))
    }

    /// Runs forward pass and returns results with `BBox` instead of `opencv::core::Rect`.
    pub fn forward_bbox(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), Error> {
        let (rects, class_ids, confidences) = self.forward(image, conf_threshold, nms_threshold)?;
        let bboxes = rects.into_iter().map(|r| r.into()).collect();
        Ok((bboxes, class_ids, confidences))
    }
}

impl ModelTrait for ModelYOLOv5OpenCV {
    fn forward(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), Error> {
        self.forward(image, conf_threshold, nms_threshold)
    }
}

impl crate::ObjectDetector for ModelYOLOv5OpenCV {
    type Input = Mat;
    type Error = Error;

    fn detect(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), Self::Error> {
        self.forward_bbox(input, conf_threshold, nms_threshold)
    }
}
