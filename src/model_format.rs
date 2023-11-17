/// Enum that should be used for constructing object detection model
/// 
/// Basic usage:
/// ```
/// use od_opencv::model_format::ModelFormat;
/// let mf = ModelFormat::Darknet;
/// ```
/// 
#[derive(PartialEq)]
pub enum ModelFormat {
    // Most of time use Darknet specification when you are using either https://github.com/AlexeyAB/darknet or https://github.com/pjreddie/darknet to train neural network of version v3, v4 or v7 (classic ones).
    Darknet,
    // When you are using other DL/ML/NN framework than Darknet one it is better to convert weights to ONNX specification and use it further
    // It is better to converte weights to ONNX specification with providing flag `opset = 12`` (So far as I tested it do not show any runtime errors) - e.g. YOLOv8 - https://github.com/ultralytics/ultralytics/issues/1097
    ONNX,
}