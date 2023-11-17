use crate::model_format::ModelFormat;
use crate::model_classic::ModelYOLOClassic;
use crate::model_ultralytics::ModelUltralyticsV8;
use opencv::{
    dnn::Net,
    Error
};

/* Shorthands to Ultralytics versions */

/// Shorthand to ModelUltralyticsV8::new_from_file
pub fn new_from_file_v8(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelUltralyticsV8, Error> {
    ModelUltralyticsV8::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelUltralyticsV8::new_from_darknet_file
pub fn new_from_darknet_file_v8(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelUltralyticsV8, Error> {
    ModelUltralyticsV8::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelUltralyticsV8::new_from_onnx_file
pub fn new_from_onnx_file_v8(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelUltralyticsV8, Error> {
    ModelUltralyticsV8::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelUltralyticsV8::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v8(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelUltralyticsV8, Error> {
    ModelUltralyticsV8::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)
}

/* Shorthands to Classic versions */

/// Shorthand to ModelYOLOClassic::new_from_file
pub fn new_from_file_v3(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_darknet_file
pub fn new_from_darknet_file_v3(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_onnx_file
pub fn new_from_onnx_file_v3(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v3(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_file
pub fn new_from_file_v4(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_darknet_file
pub fn new_from_darknet_file_v4(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_onnx_file
pub fn new_from_onnx_file_v4(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v4(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_file
pub fn new_from_file_v7(weight_file_path: &str, cfg_file_path: Option<&str>, net_size: (i32, i32), model_format: ModelFormat, backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_file(weight_file_path, cfg_file_path, net_size, model_format, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_darknet_file
pub fn new_from_darknet_file_v7(weight_file_path: &str, cfg_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_darknet_file(weight_file_path, cfg_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_onnx_file
pub fn new_from_onnx_file_v7(weight_file_path: &str, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_onnx_file(weight_file_path, net_size, backend_id, target_id, filter_classes)
}

/// Shorthand to ModelYOLOClassic::new_from_dnn
#[allow(unused_mut)]
pub fn new_from_dnn_v7(mut neural_net: Net, net_size: (i32, i32), backend_id: i32, target_id: i32, filter_classes: Vec<usize>) -> Result<ModelYOLOClassic, Error> {
    ModelYOLOClassic::new_from_dnn(neural_net, net_size, backend_id, target_id, filter_classes)
}