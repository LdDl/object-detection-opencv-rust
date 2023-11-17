use opencv::{
    prelude::NetTrait,
    prelude::NetTraitConst,
    prelude::MatTraitConst,
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
    imgproc::resize,
    imgproc::INTER_AREA,
    Error
};

use crate::model_format::ModelFormat;
use crate::utils::{
    BACKEND_TARGET_VALID,
    min_max_loc_partial
};

const YOLO_BLOB_MEAN: (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);

/// Wrapper around YOLOv8
/// See the ref. https://github.com/ultralytics/ultralytics
pub struct ModelUltralyticsV8 {
    net: Net,
    input_size: Size,
    blob_mean: Scalar,
    blob_scale: f64,
    blob_name: &'static str,
    out_layers: Vector<String>,
    filter_classes: Vec<usize>
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
    /// ```
    /// let x = 1.0;
    /// ```
    pub fn new_from_file(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Self, Error> {
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
        
        if BACKEND_TARGET_VALID.get(&backend_id).and_then(|map| map.get(&target_id)).is_none() {
            return Err(Error::new(400, format!("Combination of BACKEND '{}' and TARGET '{}' is not valid", backend_id, target_id)));
        };

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
        Ok(Self{
            net: neural_net,
            input_size: Size::new(net_size.0, net_size.1),
            blob_mean: Scalar::new(YOLO_BLOB_MEAN.0, YOLO_BLOB_MEAN.1, YOLO_BLOB_MEAN.2, YOLO_BLOB_MEAN.3),
            blob_scale: 1.0 / 255.0,
            blob_name: "",
            out_layers: out_layers,
            filter_classes: filter_classes
        })
    }
    pub fn forward(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), Error>{
        let image_width = image.cols();
        let image_height = image.rows();
        let x_factor = image_width as f32 / self.input_size.width as f32;
        let y_factor =  image_height as f32 / self.input_size.height as f32;
        let need_to_resize = image_width != self.input_size.width || image_height != self.input_size.height;
        let blobimg = match need_to_resize {
            true => {
                let mut resized_frame: Mat = Mat::default();
                resize(&image, &mut resized_frame, self.input_size, 1.0, 1.0, INTER_AREA)?;
                blob_from_image(&resized_frame, self.blob_scale, self.input_size, self.blob_mean, true, false, CV_32F)?
            },
            false => {
                blob_from_image(&image, self.blob_scale, self.input_size, self.blob_mean, true, false, CV_32F)?
            }
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
                let (_, max_score, _, max_class_index) = min_max_loc_partial(&classes_scores).unwrap();
                if max_score >= 0.25 {
                    if self.filter_classes.len() > 0 && !self.filter_classes.contains(&max_class_index) {
                        continue;
                    }
                    // Calculate box coordinates
                    let bbox: [i32; 4] = [
                        ((object_data[0] - (0.5 * object_data[2])) * x_factor).round() as i32,
                        ((object_data[1] - (0.5 * object_data[3])) * y_factor).round() as i32,
                        (*object_data[2] * x_factor).round() as i32,
                        (*object_data[3] * y_factor).round() as i32,
                    ];
                    let bbox_cv = Rect::new(bbox[0], bbox[1], bbox[2], bbox[3]);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::{
        imgcodecs::imread,
        imgcodecs::imwrite,
        imgproc::LINE_4,
        imgproc::rectangle,
        dnn::DNN_BACKEND_CUDA,
        dnn::DNN_TARGET_CUDA,
    };
    #[test]
    fn test_yolo_v8() {
        let net_width = 416;
        let net_height = 416;
        let mut model = ModelUltralyticsV8::new_from_file("pretrained/best_opset12.onnx", None, (net_width, net_height), ModelFormat::ONNX, DNN_BACKEND_CUDA, DNN_TARGET_CUDA, vec![]).unwrap();
        let mut frame = imread("images/1.png", 1).unwrap();
        let (nmsb, _, _) = model.forward(&frame, 0.25, 0.4).unwrap();

        for (_, bbox) in nmsb.iter().enumerate() {
            rectangle(&mut frame, *bbox, Scalar::from((0.0, 255.0, 0.0)), 2, LINE_4, 0).unwrap();
        }
        imwrite("images/predicted.png", &frame, &Vector::new()).unwrap();
    }
}