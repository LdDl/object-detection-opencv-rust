use std::os::raw::c_void;

use crate::bbox::BBox;
use crate::image_buffer::ImageBuffer;
use crate::postprocess::{Detection, nms, filter_by_class, detections_to_vecs, argmax};
use crate::preprocessing::{preprocess_into, PreprocessMeta};

use tensorrt_infer::{TrtEngine, TrtContext, CudaBuffer, CudaStream, BindingInfo, TrtError};

/// Error type for TensorRT model operations.
#[derive(Debug)]
pub enum TrtModelError {
    /// Error from TensorRT or CUDA
    Trt(String),
    /// Invalid model output shape
    InvalidOutputShape(String),
    /// File I/O error
    Io(std::io::Error),
}

impl std::fmt::Display for TrtModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrtModelError::Trt(e) => write!(f, "TensorRT error: {}", e),
            TrtModelError::InvalidOutputShape(s) => write!(f, "Invalid output shape: {}", s),
            TrtModelError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for TrtModelError {}

impl From<std::io::Error> for TrtModelError {
    fn from(e: std::io::Error) -> Self {
        TrtModelError::Io(e)
    }
}

impl From<TrtError> for TrtModelError {
    fn from(e: TrtError) -> Self {
        TrtModelError::Trt(e.to_string())
    }
}

/// Ultralytics YOLO model (v8, v9, v11) using TensorRT.
///
/// Loads a pre-built `.engine` file and runs inference on GPU.
/// The engine must be built separately using `trtexec`.
pub struct ModelUltralyticsRt {
    // Drop order matters: gpu_buffers → context → stream → engine
    /// GPU buffers for each binding (freed before context/engine).
    gpu_buffers: Vec<CudaBuffer>,
    /// Execution context (destroyed before engine).
    context: TrtContext,
    stream: CudaStream,
    engine: TrtEngine,

    input_width: u32,
    input_height: u32,
    class_filters: Vec<usize>,
    use_letterbox: bool,

    /// Pre-allocated NCHW f32 tensor buffer (host side).
    tensor_buf: ndarray::Array4<f32>,
    /// Pre-allocated host buffer for reading output from GPU.
    output_host_buf: Vec<f32>,

    /// Binding metadata.
    bindings: Vec<BindingInfo>,
    /// Index of the input binding.
    input_binding_idx: usize,
    /// Index of the output binding.
    output_binding_idx: usize,
    /// Output shape.
    output_shape: Vec<i32>,
}

impl ModelUltralyticsRt {
    /// Creates a new model from a TensorRT engine file.
    ///
    /// # Arguments
    /// * `engine_path` - Path to the pre-built `.engine` file
    /// * `input_size` - Model input size as (width, height)
    /// * `class_filters` - List of class indices to detect (empty for all classes)
    pub fn new_from_file(
        engine_path: &str,
        input_size: (u32, u32),
        class_filters: Vec<usize>,
    ) -> Result<Self, TrtModelError> {
        let engine = TrtEngine::from_file(engine_path)?;
        let bindings = engine.bindings();

        let input_binding_idx = bindings.iter()
            .position(|b| b.is_input)
            .ok_or_else(|| TrtModelError::Trt("No input binding found".into()))?;
        let output_binding_idx = bindings.iter()
            .position(|b| !b.is_input)
            .ok_or_else(|| TrtModelError::Trt("No output binding found".into()))?;

        let input_dims = &bindings[input_binding_idx].dims;
        if input_dims.len() != 4 || input_dims[0] != 1 || input_dims[1] != 3 {
            return Err(TrtModelError::InvalidOutputShape(
                format!("Expected input [1, 3, H, W], got {:?}", input_dims),
            ));
        }
        if input_dims[2] != input_size.1 as i32 || input_dims[3] != input_size.0 as i32 {
            return Err(TrtModelError::InvalidOutputShape(
                format!(
                    "Engine input {}x{} does not match requested {}x{}",
                    input_dims[3], input_dims[2], input_size.0, input_size.1,
                ),
            ));
        }

        let output_shape = bindings[output_binding_idx].dims.clone();

        let mut gpu_buffers = Vec::with_capacity(bindings.len());
        for binding in &bindings {
            gpu_buffers.push(CudaBuffer::new(binding.byte_size)?);
        }

        let output_num_floats = bindings[output_binding_idx].byte_size / 4;
        let output_host_buf = vec![0.0f32; output_num_floats];

        let context = engine.create_context()?;
        let stream = CudaStream::new()?;

        Ok(Self {
            engine,
            context,
            stream,
            input_width: input_size.0,
            input_height: input_size.1,
            class_filters,
            #[cfg(feature = "letterbox")]
            use_letterbox: true,
            #[cfg(not(feature = "letterbox"))]
            use_letterbox: false,
            tensor_buf: ndarray::Array4::zeros((
                1, 3, input_size.1 as usize, input_size.0 as usize,
            )),
            output_host_buf,
            gpu_buffers,
            bindings,
            input_binding_idx,
            output_binding_idx,
            output_shape,
        })
    }

    /// Enables or disables letterbox preprocessing.
    pub fn set_letterbox(&mut self, enabled: bool) {
        self.use_letterbox = enabled;
    }

    /// Returns the input size (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Runs inference on an image.
    ///
    /// # Arguments
    /// * `image` - Input image buffer
    /// * `conf_threshold` - Confidence threshold (0.0 - 1.0)
    /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
    ///
    /// # Returns
    /// Tuple of (bounding boxes, class IDs, confidence scores)
    pub fn forward(
        &mut self,
        image: &ImageBuffer,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), TrtModelError> {
        // 1. Preprocess into pre-allocated buffer
        let meta = preprocess_into(image, &mut self.tensor_buf, self.use_letterbox);

        // 2. Copy input tensor to GPU (H2D)
        let input_bytes = unsafe {
            std::slice::from_raw_parts(
                self.tensor_buf.as_ptr() as *const u8,
                self.bindings[self.input_binding_idx].byte_size,
            )
        };
        self.gpu_buffers[self.input_binding_idx].copy_from_host(input_bytes, &self.stream)?;

        // 3. Build binding pointers array
        let mut binding_ptrs: Vec<*mut c_void> = self.gpu_buffers
            .iter()
            .map(|buf| buf.as_ptr())
            .collect();

        // 4. Run inference (async on stream)
        self.context.enqueue(&mut binding_ptrs, &self.stream)?;

        // 5. Copy output from GPU to host (D2H)
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.output_host_buf.as_mut_ptr() as *mut u8,
                self.bindings[self.output_binding_idx].byte_size,
            )
        };
        self.gpu_buffers[self.output_binding_idx].copy_to_host(output_bytes, &self.stream)?;

        // 6. Synchronize stream
        self.stream.synchronize()?;

        // 7. Parse output
        let output_shape_usize: Vec<usize> = self.output_shape.iter()
            .map(|&d| d as usize)
            .collect();
        let output_view = ndarray::ArrayViewD::from_shape(
            output_shape_usize.as_slice(),
            &self.output_host_buf,
        ).map_err(|e| TrtModelError::InvalidOutputShape(format!("{}", e)))?;

        let detections = Self::parse_output_array_static(&output_view, conf_threshold, &meta)?;

        // 8. Class filter + NMS
        let class_filters = self.class_filters.clone();
        let filtered = filter_by_class(&detections, &class_filters);
        let final_detections = nms(&filtered, nms_threshold);

        Ok(detections_to_vecs(final_detections))
    }

    /// Parses the model output array into detections.
    fn parse_output_array_static(
        output: &ndarray::ArrayViewD<f32>,
        conf_threshold: f32,
        meta: &PreprocessMeta,
    ) -> Result<Vec<Detection>, TrtModelError> {
        let shape = output.shape();

        if shape.len() != 3 || shape[0] != 1 {
            return Err(TrtModelError::InvalidOutputShape(format!(
                "Expected shape [1, C, N], got {:?}",
                shape,
            )));
        }

        let num_features = shape[1];
        let num_predictions = shape[2];

        let mut detections = Vec::new();

        for i in 0..num_predictions {
            let cx = output[[0, 0, i]];
            let cy = output[[0, 1, i]];
            let w = output[[0, 2, i]];
            let h = output[[0, 3, i]];

            let class_scores: Vec<f32> = (4..num_features)
                .map(|j| output[[0, j, i]])
                .collect();

            if let Some((class_idx, max_score)) = argmax(&class_scores) {
                if max_score >= conf_threshold {
                    let (x_orig, y_orig, w_orig, h_orig) = meta.inverse_transform(cx, cy, w, h);
                    let bbox = BBox::from_center(x_orig, y_orig, w_orig, h_orig);
                    detections.push(Detection::new(bbox, class_idx, max_score));
                }
            }
        }

        Ok(detections)
    }
}

impl crate::ObjectDetector for ModelUltralyticsRt {
    type Input = ImageBuffer;
    type Error = TrtModelError;

    fn detect(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), Self::Error> {
        self.forward(input, conf_threshold, nms_threshold)
    }
}
