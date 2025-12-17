//! OpenCV DNN backend and target types.
//!
//! This module provides enums that mirror OpenCV's DNN backend and target constants,
//! allowing users to configure inference hardware without importing from opencv::dnn directly.
//!
//! # Supported combinations
//!
//! |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |  DNN_BACKEND_CUDA |
//! |------------------------|--------------------|------------------------------|--------------------|-------------------|
//! | DNN_TARGET_CPU         |                  + |                            + |                  + |                   |
//! | DNN_TARGET_OPENCL      |                  + |                            + |                  + |                   |
//! | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |                   |
//! | DNN_TARGET_MYRIAD      |                    |                            + |                    |                   |
//! | DNN_TARGET_FPGA        |                    |                            + |                    |                   |
//! | DNN_TARGET_CUDA        |                    |                              |                    |                 + |
//! | DNN_TARGET_CUDA_FP16   |                    |                              |                    |                 + |
//! | DNN_TARGET_HDDL        |                    |                            + |                    |                   |
//!
//! # Example
//!
//! ```ignore
//! use od_opencv::{Model, DnnBackend, DnnTarget};
//!
//! // CUDA inference
//! let model = Model::opencv("model.onnx", (640, 640), DnnBackend::Cuda, DnnTarget::Cuda)?;
//!
//! // CPU inference with OpenCL acceleration
//! let model = Model::opencv("model.onnx", (640, 640), DnnBackend::OpenCV, DnnTarget::OpenCL)?;
//! ```

/// DNN inference backend.
///
/// Specifies which computation backend to use for neural network inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DnnBackend {
    /// Default OpenCV backend. Most portable option.
    Default,
    /// OpenCV's built-in DNN backend.
    OpenCV,
    /// Intel Inference Engine (OpenVINO).
    InferenceEngine,
    /// Halide language backend.
    Halide,
    /// NVIDIA CUDA backend. Requires CUDA-enabled OpenCV build.
    Cuda,
}

/// DNN inference target device.
///
/// Specifies which hardware device to run inference on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DnnTarget {
    /// CPU execution.
    Cpu,
    /// OpenCL acceleration (GPU).
    OpenCL,
    /// OpenCL with FP16 precision.
    OpenCLFp16,
    /// Intel Myriad VPU.
    Myriad,
    /// FPGA device.
    Fpga,
    /// NVIDIA CUDA GPU.
    Cuda,
    /// NVIDIA CUDA GPU with FP16 precision.
    CudaFp16,
    /// Intel HDDL (High Density Deep Learning).
    Hddl,
}

impl From<DnnBackend> for i32 {
    fn from(backend: DnnBackend) -> i32 {
        use opencv::dnn;
        match backend {
            DnnBackend::Default => dnn::DNN_BACKEND_DEFAULT,
            DnnBackend::OpenCV => dnn::DNN_BACKEND_OPENCV,
            DnnBackend::InferenceEngine => dnn::DNN_BACKEND_INFERENCE_ENGINE,
            DnnBackend::Halide => dnn::DNN_BACKEND_HALIDE,
            DnnBackend::Cuda => dnn::DNN_BACKEND_CUDA,
        }
    }
}

impl From<DnnTarget> for i32 {
    fn from(target: DnnTarget) -> i32 {
        use opencv::dnn;
        match target {
            DnnTarget::Cpu => dnn::DNN_TARGET_CPU,
            DnnTarget::OpenCL => dnn::DNN_TARGET_OPENCL,
            DnnTarget::OpenCLFp16 => dnn::DNN_TARGET_OPENCL_FP16,
            DnnTarget::Myriad => dnn::DNN_TARGET_MYRIAD,
            DnnTarget::Fpga => dnn::DNN_TARGET_FPGA,
            DnnTarget::Cuda => dnn::DNN_TARGET_CUDA,
            DnnTarget::CudaFp16 => dnn::DNN_TARGET_CUDA_FP16,
            DnnTarget::Hddl => dnn::DNN_TARGET_HDDL,
        }
    }
}

impl Default for DnnBackend {
    fn default() -> Self {
        DnnBackend::OpenCV
    }
}

impl Default for DnnTarget {
    fn default() -> Self {
        DnnTarget::Cpu
    }
}
