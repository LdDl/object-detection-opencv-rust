use std::fmt;

/// Enum that should be used for constructing object detection model
/// 
/// Basic usage:
/// ```
/// use od_opencv::model_format::ModelFormat;
/// let mf = ModelFormat::Darknet;
/// ```
/// 
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum ModelFormat {
    // Most of time use Darknet specification when you are using either https://github.com/AlexeyAB/darknet or https://github.com/pjreddie/darknet to train neural network of version v3, v4 or v7 (classic ones).
    Darknet,
    // When you are using other DL/ML/NN framework than Darknet one it is better to convert weights to ONNX specification and use it further
    // It is better to converte weights to ONNX specification with providing flag `opset = 12`` (So far as I tested it do not show any runtime errors) - e.g. YOLOv8 - https://github.com/ultralytics/ultralytics/issues/1097
    ONNX,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelFormat::Darknet => write!(f, "darknet"),
            ModelFormat::ONNX => write!(f, "onnx"),
        }
    }
}

/// Enum that should be used for constructing object detection model
/// 
/// Basic usage:
/// ```
/// use od_opencv::model_format::ModelVersion;
/// let mv = ModelVersion::V3;
/// ```
/// 
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum ModelVersion {
    V3 = 3,
    V4 = 4,
    V7 = 5,
    V8 = 8,
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelVersion::V3 => write!(f, "v3"),
            ModelVersion::V4 => write!(f, "v4"),
            ModelVersion::V7 => write!(f, "v7"),
            ModelVersion::V8 => write!(f, "v8")
        }
    }
}