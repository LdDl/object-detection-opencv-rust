use super::model_format::{
    ModelFormat,
    ModelVersion
};
use super::utils::FORMAT_VERSION_VALID;
use super::model_classic::ModelYOLOClassic;
use super::model_ultralytics::ModelUltralyticsV8;
use opencv::{
    core::Mat,
    core::Rect,
    dnn::Net,
    Error
};

/// Just a trait wrapper for models
/// Should be used in scenarios when you uncertain about model type in compile time
pub trait ModelTrait {
    fn forward(&mut self, image: &Mat, conf_threshold: f32, nms_threshold: f32) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), Error>;
}

/// Creates model from file
/// 
/// Resulting structure is just a trait wrapper for ModelYOLOClassic::new_from_file and ModelUltralyticsV8::new_from_file
/// 
/// List of supported combinations model format / model version:
/// |                        | ModelFormat::Darknet | ModelFormat::ONNX |
/// |------------------------|----------------------|------------------|
/// | ModelVersion::V3       |                    + |                  |
/// | ModelVersion::V4       |                    + |                + |
/// | ModelVersion::V7       |                    + |                  |
/// | ModelVersion::V8       |                      |                + |
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
/// use opencv::dnn::{DNN_BACKEND_OPENCV, DNN_TARGET_CPU};
/// use opencv::imgcodecs::imread;
/// use od_opencv::model_format::{ModelFormat, ModelVersion};
/// use od_opencv::model::new_from_file;
/// let mf = ModelFormat::Darknet;
/// let mv = ModelVersion::V4;
/// let net_width = 416;
/// let net_height = 416;
/// let classes_labels: Vec<&str> = vec!["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];
/// let filter_classes: Vec<usize> = vec![16]; // 16-th class is 'Dog' in COCO's 80 classes. So in this case model will output only dogs if it founds them. Make it empty and you will get all detected objects.
/// let mut model = new_from_file("pretrained/yolov4-tiny.weights", Some("pretrained/yolov4-tiny.cfg"), (net_width, net_height), mf, mv, DNN_BACKEND_OPENCV, DNN_TARGET_CPU, filter_classes).unwrap();
/// let mut frame = imread("images/dog.jpg", 1).unwrap();
/// let (bboxes, class_ids, confidences) = model.forward(&frame, 0.25, 0.4).unwrap();
/// for (i, bbox) in bboxes.iter().enumerate() {
///     println!("Class: {}", classes_labels[class_ids[i]]);
///     println!("\tBounding box: {:?}", bbox);
///     println!("\tConfidences: {}", confidences[i]);
/// }
/// ```
/// 
pub fn new_from_file(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, model_version: ModelVersion, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    if FORMAT_VERSION_VALID.get(&model_format).and_then(|map| map.get(&model_version)).is_none() {
        return Err(Error::new(400, format!("Combination of model format '{}' and model version '{}' is not valid", model_format, model_version)));
    };
    match model_version {
        ModelVersion::V3 | ModelVersion::V4 | ModelVersion::V7 => {
            return Ok(Box::new(ModelYOLOClassic::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)?))
        },
        ModelVersion::V8 => {
            return Ok(Box::new(ModelUltralyticsV8::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)?))
        }
    }
}

/* Shorthands to Ultralytics versions */

/// Shorthand to ModelUltralyticsV8::new_from_file
pub fn new_from_file_v8(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelUltralyticsV8::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelUltralyticsV8::new_from_darknet_file
pub fn new_from_darknet_file_v8(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelUltralyticsV8::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelUltralyticsV8::new_from_onnx_file
pub fn new_from_onnx_file_v8(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelUltralyticsV8::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelUltralyticsV8::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v8(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelUltralyticsV8::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)?))
}

/* Shorthands to Classic versions */

/// Shorthand to ModelYOLOClassic::new_from_file
pub fn new_from_file_v3(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_darknet_file
pub fn new_from_darknet_file_v3(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_onnx_file
pub fn new_from_onnx_file_v3(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v3(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_file
pub fn new_from_file_v4(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_darknet_file
pub fn new_from_darknet_file_v4(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_onnx_file
pub fn new_from_onnx_file_v4(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v4(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_file
pub fn new_from_file_v7(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_darknet_file
pub fn new_from_darknet_file_v7(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_onnx_file
pub fn new_from_onnx_file_v7(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)?))
}

/// Shorthand to ModelYOLOClassic::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v7(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<Box<dyn ModelTrait>, Error> {
    Ok(Box::new(ModelYOLOClassic::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)?))
}