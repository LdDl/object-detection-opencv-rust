//! Model factory for creating object detection models.
//!
//! This module provides a unified `Model` factory with static methods
//! for creating models with different backends. Each method returns
//! the concrete model type (zero-cost, no dynamic dispatch).
//!
//! # Example
//!
//! ```ignore
//! use od_opencv::{Model, DnnBackend, DnnTarget};
//!
//! // ORT backend (CPU)
//! let model = Model::ort("yolov8n.onnx", (640, 640))?;
//!
//! // ORT backend with CUDA
//! let model = Model::ort_cuda("yolov8n.onnx", (640, 640))?;
//!
//! // OpenCV backend for Ultralytics models (CUDA)
//! let model = Model::opencv("yolov8n.onnx", (640, 640), DnnBackend::Cuda, DnnTarget::Cuda)?;
//!
//! // OpenCV backend for Darknet models (CUDA)
//! let model = Model::darknet("yolov4.cfg", "yolov4.weights", (416, 416), DnnBackend::Cuda, DnnTarget::Cuda)?;
//! ```

/// Factory for creating object detection models.
///
/// This is a zero-sized type that serves as a namespace for model constructors.
/// Each method returns the concrete model type, enabling full compiler optimization.
pub struct Model;

// ============================================================================
// ORT Backend
// ============================================================================

#[cfg(feature = "ort-backend")]
impl Model {
    /// Creates a new Ultralytics YOLO model (v8/v9/v11) using ONNX Runtime (CPU).
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    ///
    /// # Example
    /// ```ignore
    /// let mut model = Model::ort("yolov8n.onnx", (640, 640))?;
    /// ```
    pub fn ort(
        model_path: &str,
        input_size: (u32, u32),
    ) -> Result<crate::backend_ort::ModelUltralyticsOrt, crate::backend_ort::OrtModelError> {
        crate::backend_ort::ModelUltralyticsOrt::new_from_file(model_path, input_size, vec![])
    }

    /// Creates a new Ultralytics YOLO model with class filtering using ONNX Runtime (CPU).
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `class_filters` - List of class indices to detect (empty for all classes)
    pub fn ort_filtered(
        model_path: &str,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Result<crate::backend_ort::ModelUltralyticsOrt, crate::backend_ort::OrtModelError> {
        crate::backend_ort::ModelUltralyticsOrt::new_from_file(model_path, input_size, class_filters)
    }
}

#[cfg(feature = "ort-cuda-backend")]
impl Model {
    /// Creates a new Ultralytics YOLO model using ONNX Runtime with CUDA acceleration.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    ///
    /// # Example
    /// ```ignore
    /// let mut model = Model::ort_cuda("yolov8n.onnx", (640, 640))?;
    /// ```
    pub fn ort_cuda(
        model_path: &str,
        input_size: (u32, u32),
    ) -> Result<crate::backend_ort::ModelUltralyticsOrt, crate::backend_ort::OrtModelError> {
        crate::backend_ort::ModelUltralyticsOrt::new_from_file_cuda(model_path, input_size, vec![])
    }

    /// Creates a new Ultralytics YOLO model with class filtering using ONNX Runtime with CUDA.
    pub fn ort_cuda_filtered(
        model_path: &str,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Result<crate::backend_ort::ModelUltralyticsOrt, crate::backend_ort::OrtModelError> {
        crate::backend_ort::ModelUltralyticsOrt::new_from_file_cuda(model_path, input_size, class_filters)
    }
}

#[cfg(feature = "ort-tensorrt-backend")]
impl Model {
    /// Creates a new Ultralytics YOLO model using ONNX Runtime with TensorRT acceleration.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    ///
    /// # Example
    /// ```ignore
    /// let mut model = Model::ort_tensorrt("yolov8n.onnx", (640, 640))?;
    /// ```
    pub fn ort_tensorrt(
        model_path: &str,
        input_size: (u32, u32),
    ) -> Result<crate::backend_ort::ModelUltralyticsOrt, crate::backend_ort::OrtModelError> {
        crate::backend_ort::ModelUltralyticsOrt::new_from_file_tensorrt(model_path, input_size, vec![])
    }

    /// Creates a new Ultralytics YOLO model with class filtering using TensorRT.
    pub fn ort_tensorrt_filtered(
        model_path: &str,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Result<crate::backend_ort::ModelUltralyticsOrt, crate::backend_ort::OrtModelError> {
        crate::backend_ort::ModelUltralyticsOrt::new_from_file_tensorrt(model_path, input_size, class_filters)
    }
}

// ============================================================================
// OpenCV Backend
// ============================================================================

#[cfg(feature = "opencv-backend")]
impl Model {
    /// Creates a new Ultralytics YOLO model (v8/v9/v11) using OpenCV DNN.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `backend` - DNN backend (e.g., `DnnBackend::Cuda`, `DnnBackend::OpenCV`)
    /// * `target` - DNN target device (e.g., `DnnTarget::Cuda`, `DnnTarget::Cpu`)
    ///
    /// # Example
    /// ```ignore
    /// use od_opencv::{Model, DnnBackend, DnnTarget};
    ///
    /// // CUDA inference
    /// let mut model = Model::opencv("yolov8n.onnx", (640, 640), DnnBackend::Cuda, DnnTarget::Cuda)?;
    ///
    /// // CPU inference
    /// let mut model = Model::opencv("yolov8n.onnx", (640, 640), DnnBackend::OpenCV, DnnTarget::Cpu)?;
    /// ```
    pub fn opencv(
        model_path: &str,
        input_size: (i32, i32),
        backend: crate::dnn_backend::DnnBackend,
        target: crate::dnn_backend::DnnTarget,
    ) -> Result<crate::backend_opencv::model_ultralytics::ModelUltralyticsV8, opencv::Error> {
        crate::backend_opencv::model_ultralytics::ModelUltralyticsV8::new_from_onnx_file(
            model_path,
            input_size,
            backend.into(),
            target.into(),
            vec![],
        )
    }

    /// Creates a new Ultralytics YOLO model with class filtering using OpenCV DNN.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `backend` - DNN backend
    /// * `target` - DNN target device
    /// * `class_filters` - List of class indices to detect (empty for all)
    pub fn opencv_filtered(
        model_path: &str,
        input_size: (i32, i32),
        backend: crate::dnn_backend::DnnBackend,
        target: crate::dnn_backend::DnnTarget,
        class_filters: Vec<usize>,
    ) -> Result<crate::backend_opencv::model_ultralytics::ModelUltralyticsV8, opencv::Error> {
        crate::backend_opencv::model_ultralytics::ModelUltralyticsV8::new_from_onnx_file(
            model_path,
            input_size,
            backend.into(),
            target.into(),
            class_filters,
        )
    }

    /// Creates a new classic YOLO model (v3/v4/v7) from Darknet files using OpenCV DNN.
    ///
    /// # Arguments
    /// * `cfg_path` - Path to the Darknet .cfg file
    /// * `weights_path` - Path to the Darknet .weights file
    /// * `input_size` - Model input size as (width, height)
    /// * `backend` - DNN backend
    /// * `target` - DNN target device
    ///
    /// # Example
    /// ```ignore
    /// use od_opencv::{Model, DnnBackend, DnnTarget};
    ///
    /// let mut model = Model::darknet(
    ///     "yolov4.cfg",
    ///     "yolov4.weights",
    ///     (416, 416),
    ///     DnnBackend::Cuda,
    ///     DnnTarget::Cuda
    /// )?;
    /// ```
    pub fn darknet(
        cfg_path: &str,
        weights_path: &str,
        input_size: (i32, i32),
        backend: crate::dnn_backend::DnnBackend,
        target: crate::dnn_backend::DnnTarget,
    ) -> Result<crate::backend_opencv::model_classic::ModelYOLOClassic, opencv::Error> {
        crate::backend_opencv::model_classic::ModelYOLOClassic::new_from_darknet_file(
            weights_path,
            cfg_path,
            input_size,
            backend.into(),
            target.into(),
            vec![],
        )
    }

    /// Creates a new classic YOLO model with class filtering from Darknet files using OpenCV DNN.
    ///
    /// # Arguments
    /// * `cfg_path` - Path to the Darknet .cfg file
    /// * `weights_path` - Path to the Darknet .weights file
    /// * `input_size` - Model input size as (width, height)
    /// * `backend` - DNN backend
    /// * `target` - DNN target device
    /// * `class_filters` - List of class indices to detect (empty for all)
    pub fn darknet_filtered(
        cfg_path: &str,
        weights_path: &str,
        input_size: (i32, i32),
        backend: crate::dnn_backend::DnnBackend,
        target: crate::dnn_backend::DnnTarget,
        class_filters: Vec<usize>,
    ) -> Result<crate::backend_opencv::model_classic::ModelYOLOClassic, opencv::Error> {
        crate::backend_opencv::model_classic::ModelYOLOClassic::new_from_darknet_file(
            weights_path,
            cfg_path,
            input_size,
            backend.into(),
            target.into(),
            class_filters,
        )
    }

    /// Creates a new classic YOLO model (v3/v4/v7) from ONNX file using OpenCV DNN.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `backend` - DNN backend
    /// * `target` - DNN target device
    ///
    /// # Example
    /// ```ignore
    /// use od_opencv::{Model, DnnBackend, DnnTarget};
    ///
    /// let mut model = Model::classic_onnx(
    ///     "yolov4-tiny.onnx",
    ///     (416, 416),
    ///     DnnBackend::Cuda,
    ///     DnnTarget::Cuda
    /// )?;
    /// ```
    pub fn classic_onnx(
        model_path: &str,
        input_size: (i32, i32),
        backend: crate::dnn_backend::DnnBackend,
        target: crate::dnn_backend::DnnTarget,
    ) -> Result<crate::backend_opencv::model_classic::ModelYOLOClassic, opencv::Error> {
        crate::backend_opencv::model_classic::ModelYOLOClassic::new_from_onnx_file(
            model_path,
            input_size,
            backend.into(),
            target.into(),
            vec![],
        )
    }

    /// Creates a new classic YOLO model with class filtering from ONNX using OpenCV DNN.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `input_size` - Model input size as (width, height)
    /// * `backend` - DNN backend
    /// * `target` - DNN target device
    /// * `class_filters` - List of class indices to detect (empty for all)
    pub fn classic_onnx_filtered(
        model_path: &str,
        input_size: (i32, i32),
        backend: crate::dnn_backend::DnnBackend,
        target: crate::dnn_backend::DnnTarget,
        class_filters: Vec<usize>,
    ) -> Result<crate::backend_opencv::model_classic::ModelYOLOClassic, opencv::Error> {
        crate::backend_opencv::model_classic::ModelYOLOClassic::new_from_onnx_file(
            model_path,
            input_size,
            backend.into(),
            target.into(),
            class_filters,
        )
    }
}
