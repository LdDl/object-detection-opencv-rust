use opencv::{
    prelude::NetTrait,
    prelude::NetTraitConst,
    prelude::MatTraitConst,
    prelude::MatTraitConstManual,
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
use crate::utils::BACKEND_TARGET_VALID;

const YOLO_BLOB_MEAN: (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);

/// Wrapper around "classic" versions of YOLO (v3, v4 and v7 are supported currently)
/// "Classic" - because they are developed historically by AlexeyAB and pjreddie (see the ref. https://github.com/AlexeyAB/darknet and https://github.com/pjreddie/darknet)
pub struct ModelYOLOClassic {
    net: Net,
    input_size: Size,
    blob_mean: Scalar,
    blob_scale: f64,
    blob_name: &'static str,
    out_layers: Vector<String>,
    filter_classes: Vec<usize>
}

impl ModelYOLOClassic {
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
            return ModelYOLOClassic::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)
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

        ModelYOLOClassic::new_from_darknet_file(weight_file_path, cfg, net_size, backend_id, target_id, filter_classes)
    }
    /// Reads file in Darknet specification and prepares model
    pub fn new_from_darknet_file(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Self, Error> {
        ModelYOLOClassic::new_from_dnn(read_net(weight_file_path, cfg_file_path, "Darknet")?, net_size, backend_id, target_id, filter_classes)
    }
    /// Reads file in ONNX specification and prepares model
    pub fn new_from_onnx_file(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Self, Error> {
        ModelYOLOClassic::new_from_dnn(read_net_from_onnx(weight_file_path)?, net_size, backend_id, target_id, filter_classes)
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
        let image_width_f32 = image_width as f32;
        let image_height_f32 = image_height as f32;
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

        // Specific to YOLOv3, YOLOv4, YOLOv7 reading detections vector
        for layer in detections {
            let num_boxes = layer.rows();
            for index in 0..num_boxes {
                let pred = layer.row(index)?;
                let detection = pred.data_typed::<f32>()?;
                let (center_x, center_y, width, height, confidence) = match &detection[0..5] {
                    &[a,b,c,d,e] => (a * image_width_f32, b * image_height_f32, c * image_width_f32, d * image_height_f32, e),
                    _ => {
                        return Err(Error::new(500, "Can't extract (center_x, center_y, width, height, confidence) from detection vector"))
                    }
                };
                let detected_classes = &detection[5..];
                if confidence > conf_threshold {
                    let mut class_index = -1;
                    let mut score = 0.0;
                    for (idx, &val) in detected_classes.iter().enumerate() {
                        if val > score {
                            class_index = idx as i32;
                            score = val;
                        }
                    }
                    if class_index > -1 && score > 0. {
                        let class_id = class_index as usize;
                        if self.filter_classes.len() > 0 && !self.filter_classes.contains(&class_id) {
                            continue;
                        }
                        let left = center_x - width / 2.0;
                        let top = center_y - height / 2.0;
                        let bbox = Rect::new(left.floor() as i32, top.floor() as i32, width as i32, height as i32);
                        class_ids.push(class_id);
                        confidences.push(confidence);
                        bboxes.push(bbox);
                    }
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
    fn test_yolo_classic() {
        let net_width = 416;
        let net_height = 416;
        let mut model = ModelYOLOClassic::new_from_file("pretrained/tinyv3-vehicles_best.weights", Some("pretrained/tinyv3-vehicles.cfg"), (net_width, net_height), ModelFormat::Darknet, DNN_BACKEND_CUDA, DNN_TARGET_CUDA, vec![]).unwrap();
        let mut frame = imread("images/1.png", 1).unwrap();
        let (nmsb, _, _) = model.forward(&frame, 0.25, 0.4).unwrap();

        for (_, bbox) in nmsb.iter().enumerate() {
            rectangle(&mut frame, *bbox, Scalar::from((0.0, 255.0, 0.0)), 2, LINE_4, 0).unwrap();
        }
        imwrite("images/predicted_classic.png", &frame, &Vector::new()).unwrap();
    }
}
