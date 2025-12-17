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
    dnn::read_net,
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

use super::model_format::ModelFormat;
use super::model::ModelTrait;
use super::utils::{
    BACKEND_TARGET_VALID,
    min_max_loc_partial
};

const YOLO_BLOB_MEAN: (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);

/// Wrapper around YOLOv8
/// See the ref. https://github.com/ultralytics/ultralytics
pub struct ModelUltralyticsV8 {
    // Underlying OpenCV's DNN Net implementation
    net: Net,
    // Input size a.k.a network size (width and height of input). It is usefull when calculating relative bounding box to size of source image or resizing image to proper (width, height)
    input_size: Size,
    // Blob's scale for OpenCV's blob. For YOLO it is just (0.0 - red channel, 0.0 - blue, 0.0 - green, 0.0 - alpha) most of time (if it is not, than it is needed to make this field adjustable)
    blob_mean: Scalar,
    // Blob's scale for OpenCV's blob. For YOLO it is just 1/255 (~ 0.0039) most of time (if it is not, than it is needed to make this field adjustable)
    blob_scale: f64,
    // Blob's name basically for OpenCV's blob. For YOLO it is just an empty string most of time (if it is not, than it is needed to make this field adjustable)
    blob_name: &'static str,
    // Layers to aggregate results from (for some models there could me multiple YOLO layers)
    out_layers: Vector<String>,
    // Set of classes which will be used to filter detections
    filter_classes: Vec<usize>,
    // Reusable buffer for letterbox resize (avoids allocation per frame)
    #[cfg(feature = "letterbox")]
    letterbox_resized: Mat,
    // Reusable buffer for letterbox padding (avoids allocation per frame)
    #[cfg(feature = "letterbox")]
    letterbox_padded: Mat,
}

impl ModelUltralyticsV8 {
    /// Read file for specified model format, BACKEND and TARGET combo and then prepares model.
    /// 
    /// List of supported combinations backend / target:
	/// |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |  DNN_BACKEND_CUDA |
	/// |------------------------|--------------------|------------------------------|--------------------|-------------------|
	/// | DNN_TARGET_CPU         |                  + |                            + |                  + |                   |
	/// | DNN_TARGET_OPENCL      |                  + |                            + |                  + |                   |
	/// | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |                   |
	/// | DNN_TARGET_MYRIAD      |                    |                            + |                    |                   |
	/// | DNN_TARGET_FPGA        |                    |                            + |                    |                   |
	/// | DNN_TARGET_CUDA        |                    |                              |                    |                 + |
	/// | DNN_TARGET_CUDA_FP16   |                    |                              |                    |                 + |
	/// | DNN_TARGET_HDDL        |                    |                            + |                    |                   |
    /// 
    /// Basic usage:
    ///
    /// ```ignore
    /// use opencv::dnn::{DNN_BACKEND_OPENCV, DNN_TARGET_CPU};
    /// use opencv::imgcodecs::imread;
    /// use od_opencv::model_format::ModelFormat;
    /// use od_opencv::model_ultralytics::ModelUltralyticsV8;
    /// let mf = ModelFormat::ONNX;
    /// let net_width = 640;
    /// let net_height = 640;
    /// let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
    /// let filter_classes: Vec<usize> = vec![16]; // 16-th class is 'Dog' in COCO's 80 classes. So in this case model will output only dogs if it founds them. Make it empty and you will get all detected objects.
    /// let mut model = ModelUltralyticsV8::new_from_file("pretrained/yolov8n.onnx", None, (net_width, net_height), mf, DNN_BACKEND_OPENCV, DNN_TARGET_CPU, filter_classes).unwrap();
    /// let mut frame = imread("images/dog.jpg", 1).unwrap();
    /// let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();
    /// for (i, bbox) in bboxes.iter().enumerate() {
    ///     println!("Class: {}", classes_labels[class_ids[i]]);
    ///     println!("\tBounding box: {:?}", bbox);
    ///     println!("\tConfidences: {}", confidences[i]);
    /// }
    /// ```
    /// 
    pub fn new_from_file(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Self, Error> {
        if BACKEND_TARGET_VALID.get(&backend_id).and_then(|map| map.get(&target_id)).is_none() {
            return Err(Error::new(400, format!("Combination of BACKEND '{}' and TARGET '{}' is not valid", backend_id, target_id)));
        };

        if model_format == ModelFormat::ONNX {
            return ModelUltralyticsV8::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)
        }
        let cfg = match cfg_file_path {
            Some(s) => {
                if s == "" {
                    return Err(Error::new(400, "Empty configuration file path"))
                }
                Ok(s)
            },
            None => { Err(Error::new(400, "No configuration file path has been provided")) }
        }?;

        ModelUltralyticsV8::new_from_darknet_file(weight_file_path, cfg, net_size, backend_id, target_id, filter_classes)
    }
    /// Reads file in Darknet specification and prepares model
    pub fn new_from_darknet_file(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Self, Error> {
        ModelUltralyticsV8::new_from_dnn(read_net(weight_file_path, cfg_file_path, "Darknet")?, net_size, backend_id, target_id, filter_classes)
    }
    /// Reads file in ONNX specification and prepares model
    pub fn new_from_onnx_file(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Self, Error> {
        ModelUltralyticsV8::new_from_dnn(read_net_from_onnx(weight_file_path)?, net_size, backend_id, target_id, filter_classes)
    }
    /// Prepares model from OpenCV's DNN neural network
    pub fn new_from_dnn(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Self, Error> {
        neural_net.set_preferable_backend(backend_id)?;
        neural_net.set_preferable_target(target_id)?;
        let out_layers = neural_net.get_unconnected_out_layers_names()?;

        #[cfg(feature = "letterbox")]
        // Pre-allocate letterbox_padded with known target size (width x height, 3 channels)
        let letterbox_padded = Mat::new_rows_cols_with_default(
            net_size.1,  // height
            net_size.0,  // width
            CV_8UC3,
            Scalar::new(114.0, 114.0, 114.0, 0.0)  // gray padding color
        )?;

        Ok(Self{
            net: neural_net,
            input_size: Size::new(net_size.0, net_size.1),
            blob_mean: Scalar::new(YOLO_BLOB_MEAN.0, YOLO_BLOB_MEAN.1, YOLO_BLOB_MEAN.2, YOLO_BLOB_MEAN.3),
            blob_scale: 1.0 / 255.0,
            blob_name: "",
            out_layers,
            filter_classes,
            #[cfg(feature = "letterbox")]
            letterbox_resized: Mat::default(),  // size varies with input aspect ratio
            #[cfg(feature = "letterbox")]
            letterbox_padded,
        })
    }
    pub fn forward(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), Error>{
        let image_width = image.cols();
        let image_height = image.rows();

        // Preprocessing and coordinate conversion factors depend on feature flag
        #[cfg(feature = "letterbox")]
        let (blobimg, scale, pad_left, pad_top) = {
            // Letterbox preprocessing: resize maintaining aspect ratio, then pad
            let scale = f32::min(
                self.input_size.width as f32 / image_width as f32,
                self.input_size.height as f32 / image_height as f32
            );
            let new_width = (image_width as f32 * scale).round() as i32;
            let new_height = (image_height as f32 * scale).round() as i32;
            let pad_left = (self.input_size.width - new_width) / 2;
            let pad_top = (self.input_size.height - new_height) / 2;

            let blob = if image_width != self.input_size.width || image_height != self.input_size.height {
                // Resize maintaining aspect ratio (reuses buffer if size matches)
                resize(&image, &mut self.letterbox_resized, Size::new(new_width, new_height), 0.0, 0.0, INTER_LINEAR)?;
                // Pad to target size with gray (114, 114, 114)
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
                // Size(0,0) tells OpenCV to use padded's dimensions as-is
                // See: https://github.com/opencv/opencv/blob/4.x/samples/dnn/object_detection.cpp#L54
                blob_from_image(&self.letterbox_padded, self.blob_scale, Size::new(0, 0), self.blob_mean, true, false, CV_32F)?
            } else {
                blob_from_image(&image, self.blob_scale, self.input_size, self.blob_mean, true, false, CV_32F)?
            };
            (blob, scale, pad_left, pad_top)
        };

        #[cfg(not(feature = "letterbox"))]
        let (blobimg, scale_x, scale_y) = {
            // Stretch preprocessing: direct resize to input_size (faster but may distort)
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

        // Specific to YOLOv8 reading detections vector
        // See the ref. https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py#L65
        for layer in detections {
            let mat_size = layer.mat_size();
            let cols = mat_size[1];
            let rows: i32 = mat_size[2];
            for i in 0..rows{
                // Access elements along the second dimension
                let object_data: Vec<&f32> = (0..cols)
                    .map(|j| layer.at_3d::<f32>(0, j, i).unwrap())
                    .collect();
                // Access elements as if transposed
                let classes_scores: Vec<&f32> = object_data[4..].to_vec();
                // Find min and max scores and locations
                let (_, max_score, _, max_class_index) = match min_max_loc_partial(&classes_scores) {
                    Some((a, b, c, d)) => {
                        (a, b, c, d)
                    },
                    None => {
                        return Err(Error::new(500, "Can't execute min_max_loc_partial due the class_scores is an empty array"));
                    }
                };
                if max_score >= 0.25 {
                    if self.filter_classes.len() > 0 && !self.filter_classes.contains(&max_class_index) {
                        continue;
                    }

                    // Coordinate conversion: model outputs pixel coordinates in input space
                    #[cfg(feature = "letterbox")]
                    let (x_center, y_center, width, height) = {
                        // Letterbox: subtract padding offset, divide by scale factor
                        (
                            (*object_data[0] - pad_left as f32) / scale,
                            (*object_data[1] - pad_top as f32) / scale,
                            *object_data[2] / scale,
                            *object_data[3] / scale,
                        )
                    };

                    #[cfg(not(feature = "letterbox"))]
                    let (x_center, y_center, width, height) = {
                        // Stretch: scale back from input_size to original image size
                        (
                            *object_data[0] * scale_x,
                            *object_data[1] * scale_y,
                            *object_data[2] * scale_x,
                            *object_data[3] * scale_y,
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
                    confidences.push(max_score);
                    class_ids.push(max_class_index);
                }
            }
        }
        // Run NMS on collected detections to filter duplicates and overlappings
        let mut indices = Vector::<i32>::new();
        nms_boxes(&bboxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1.0, 0)?;

        let mut nms_bboxes = vec![];
        let mut nms_classes_ids = vec![];
        let mut nms_confidences = vec![];

        let indices_vec = indices.to_vec();
        let mut bboxes = bboxes.to_vec();
        nms_bboxes.extend(bboxes.drain(..)
            .enumerate()
            .filter_map(|(idx, item)| if indices_vec.contains(&(idx as i32)) {Some(item)} else {None}));

        nms_classes_ids.extend(class_ids.drain(..)
            .enumerate()
            .filter_map(|(idx, item)| if indices_vec.contains(&(idx as i32)) {Some(item)} else {None}));

        nms_confidences.extend(confidences.to_vec().drain(..)
            .enumerate()
            .filter_map(|(idx, item)| if indices_vec.contains(&(idx as i32)) {Some(item)} else {None}));

        Ok((nms_bboxes, nms_classes_ids, nms_confidences))
    }

    /// Runs forward pass and returns results with `BBox` instead of `opencv::core::Rect`.
    ///
    /// This is a convenience method for users who prefer the backend-agnostic `BBox` type.
    /// Internally calls `forward()` and converts the results.
    pub fn forward_bbox(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), Error> {
        let (rects, class_ids, confidences) = self.forward(image, conf_threshold, nms_threshold)?;
        let bboxes = rects.into_iter().map(|r| r.into()).collect();
        Ok((bboxes, class_ids, confidences))
    }
}

impl ModelTrait for ModelUltralyticsV8 {
    fn forward(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), Error>{
        self.forward(image, conf_threshold, nms_threshold)
    }
}