//! ArcFace face recognition model using ONNX Runtime.
//!
//! Supports InsightFace model zoo models (w600k_mbf, w600k_r50, etc.).
//! Input: [1, 3, 112, 112] float32, RGB
//! Output: [1, 512] float32 embedding, L2-normalized
//!
//! Normalization is configurable:
//! - MobileFaceNet (w600k_mbf): `(pixel - 127.5) / 127.5` => [-1, 1]
//! - ResNet50 (w600k_r50): `pixel / 255.0` => [0, 1]

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::inputs;
use ort::value::TensorRef;

use ndarray::Array4;

use crate::image_buffer::ImageBuffer;

use super::OrtModelError;

/// Input normalization for ArcFace models.
///
/// Different InsightFace models expect different pixel normalization:
/// - `MobileFaceNet`: `(pixel - 127.5) / 127.5` => [-1, 1] (w600k_mbf)
/// - `ResNet`: `pixel / 255.0` => [0, 1] (w600k_r50)
///
/// Custom normalization is also supported via `Custom { mean, scale }`
/// where the formula is `(pixel - mean) * scale`.
#[derive(Debug, Clone, Copy)]
pub enum ArcFaceNorm {
    /// (pixel - 127.5) / 127.5 => [-1, 1]. Default for w600k_mbf (MobileFaceNet).
    MobileFaceNet,
    /// pixel / 255.0 => [0, 1]. Default for w600k_r50 (ResNet50).
    ResNet,
    /// (pixel - mean) * scale
    Custom { mean: f32, scale: f32 },
}

impl Default for ArcFaceNorm {
    fn default() -> Self {
        ArcFaceNorm::MobileFaceNet
    }
}

impl ArcFaceNorm {
    #[inline]
    fn normalize(self, pixel: f32) -> f32 {
        match self {
            ArcFaceNorm::MobileFaceNet => (pixel - 127.5) / 127.5,
            ArcFaceNorm::ResNet => pixel / 255.0,
            ArcFaceNorm::Custom { mean, scale } => (pixel - mean) * scale,
        }
    }
}

/// ArcFace face recognition model using ONNX Runtime.
///
/// Takes a 112x112 aligned face crop and produces a 512-dimensional
/// L2-normalized embedding vector.
pub struct ModelArcFaceOrt {
    session: Session,
    tensor_buf: Array4<f32>,
    input_name: String,
    output_name: String,
    norm: ArcFaceNorm,
    input_size: u32,
}

impl ModelArcFaceOrt {
    /// Creates a new ArcFace model from an ONNX file (CPU).
    ///
    /// Uses `MobileFaceNet` normalization by default ([-1, 1]).
    /// For ResNet50 models (w600k_r50), use [`new_from_file_with_norm`].
    ///
    /// # Example
    /// ```ignore
    /// let model = ModelArcFaceOrt::new_from_file("w600k_mbf.onnx")?;
    /// ```
    pub fn new_from_file(model_path: &str) -> Result<Self, OrtModelError> {
        Self::new_from_file_with_norm(model_path, ArcFaceNorm::default())
    }

    /// Creates a new ArcFace model from an ONNX file (CPU) with explicit normalization.
    ///
    /// # Example
    /// ```ignore
    /// // ResNet50 model
    /// let model = ModelArcFaceOrt::new_from_file_with_norm(
    ///     "w600k_r50.onnx",
    ///     ArcFaceNorm::ResNet,
    /// )?;
    /// ```
    pub fn new_from_file_with_norm(model_path: &str, norm: ArcFaceNorm) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Self::from_session(session, norm)
    }

    /// Creates a new ArcFace model with CUDA acceleration.
    ///
    /// Uses `MobileFaceNet` normalization by default.
    #[cfg(feature = "ort-cuda-backend")]
    pub fn new_from_file_cuda(model_path: &str) -> Result<Self, OrtModelError> {
        Self::new_from_file_cuda_with_norm(model_path, ArcFaceNorm::default())
    }

    /// Creates a new ArcFace model with CUDA acceleration and explicit normalization.
    #[cfg(feature = "ort-cuda-backend")]
    pub fn new_from_file_cuda_with_norm(model_path: &str, norm: ArcFaceNorm) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Self::from_session(session, norm)
    }

    /// Creates a new ArcFace model with TensorRT acceleration via ORT.
    ///
    /// Uses `MobileFaceNet` normalization by default.
    #[cfg(feature = "ort-tensorrt-backend")]
    pub fn new_from_file_tensorrt(model_path: &str) -> Result<Self, OrtModelError> {
        Self::new_from_file_tensorrt_with_norm(model_path, ArcFaceNorm::default())
    }

    /// Creates a new ArcFace model with TensorRT acceleration and explicit normalization.
    #[cfg(feature = "ort-tensorrt-backend")]
    pub fn new_from_file_tensorrt_with_norm(model_path: &str, norm: ArcFaceNorm) -> Result<Self, OrtModelError> {
        let session = Session::builder()?
            .with_execution_providers([ort::execution_providers::TensorRTExecutionProvider::default().build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Self::from_session(session, norm)
    }

    fn from_session(session: Session, norm: ArcFaceNorm) -> Result<Self, OrtModelError> {
        let inputs = session.inputs();
        if inputs.is_empty() {
            return Err(OrtModelError::InvalidOutputShape("ArcFace model has no inputs".into()));
        }
        let input_shape = inputs[0].dtype().tensor_shape()
            .ok_or_else(|| OrtModelError::InvalidOutputShape("Cannot read input tensor shape".into()))?;
        if input_shape.len() != 4 {
            return Err(OrtModelError::InvalidOutputShape(
                format!("Expected 4D input [1,3,H,W], got {}D", input_shape.len()),
            ));
        }
        if input_shape[1] != 3 || input_shape[2] != input_shape[3] {
            return Err(OrtModelError::InvalidOutputShape(
                format!("Expected square input [1,3,S,S], got {:?}", input_shape),
            ));
        }
        let input_size = input_shape[2] as u32;

        let input_name = inputs[0].name().to_string();

        let outputs_info = session.outputs();
        if outputs_info.is_empty() {
            return Err(OrtModelError::InvalidOutputShape("ArcFace model has no outputs".into()));
        }
        let output_name = outputs_info[0].name().to_string();

        let s = input_size as usize;
        let tensor_buf = Array4::<f32>::zeros((1, 3, s, s));
        Ok(Self { session, tensor_buf, input_name, output_name, norm, input_size })
    }

    /// Returns the expected aligned face input size (square side, e.g. 112).
    pub fn input_size(&self) -> u32 {
        self.input_size
    }

    /// Extracts a 512-dimensional embedding from an aligned 112x112 face.
    ///
    /// The input image must be a 112x112 RGB crop produced by [`crate::face_alignment::align_face`].
    /// Normalization is applied according to the [`ArcFaceNorm`] set during construction.
    ///
    /// # Arguments
    /// * `aligned_face` - A 112x112 RGB `ImageBuffer`
    ///
    /// # Returns
    /// L2-normalized 512-dimensional embedding vector.
    pub fn forward(&mut self, aligned_face: &ImageBuffer) -> Result<[f32; 512], OrtModelError> {
        let (h, w, _) = aligned_face.shape();
        let s = self.input_size as usize;
        if h != s || w != s {
            return Err(OrtModelError::InvalidOutputShape(
                format!("ArcFace requires {s}x{s} input, got {}x{}", w, h),
            ));
        }

        // Fill tensor buffer: HWC RGB u8 -> NCHW RGB f32, normalized per ArcFaceNorm
        let src = aligned_face.as_array();
        let norm = self.norm;
        for c in 0..3 {
            for y in 0..s {
                for x in 0..s {
                    self.tensor_buf[[0, c, y, x]] =
                        norm.normalize(src[[y, x, c]] as f32);
                }
            }
        }

        let outputs = self.session.run(
            inputs![self.input_name.as_str() => TensorRef::from_array_view(&self.tensor_buf)?]
        )?;

        let embedding_view = outputs[self.output_name.as_str()].try_extract_array::<f32>()?;
        let embedding_slice = embedding_view.as_standard_layout();
        let flat = embedding_slice.as_slice()
            .ok_or_else(|| OrtModelError::InvalidOutputShape("Cannot get embedding slice".into()))?;

        if flat.len() < 512 {
            return Err(OrtModelError::InvalidOutputShape(
                format!("Expected 512-dim embedding, got {}", flat.len()),
            ));
        }

        // L2 normalize
        let mut embedding = [0.0f32; 512];
        let mut norm_sq = 0.0f32;
        for i in 0..512 {
            embedding[i] = flat[i];
            norm_sq += flat[i] * flat[i];
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        Ok(embedding)
    }
}
