//! YuNet face detection model using TensorRT.
//!
//! Model: face_detection_yunet_2023mar (OpenCV Zoo)
//! Input: [1, 3, H, W] float32, BGR, [0..255]
//! Outputs: 12 bindings (4 per stride 8/16/32): cls, obj, bbox, kps

use std::os::raw::c_void;

use tensorrt_infer::{TrtEngine, TrtContext, CudaBuffer, CudaStream, BindingInfo, TrtError};

use crate::face_detection::{FaceDetection, FaceDetector, STRIDES, decode_yunet_stride, nms_faces};
use crate::image_buffer::ImageBuffer;
use crate::preprocessing::{PreprocessMeta, preprocess_into_bgr_letterbox};

use super::TrtModelError;

/// Per-stride output binding indices and host buffer info.
struct StrideBindings {
    stride: u32,
    cls_idx: usize,
    obj_idx: usize,
    bbox_idx: usize,
    kps_idx: usize,
}

/// YuNet face detection model using TensorRT.
///
/// Loads a pre-built `.engine` file with 12 output bindings.
pub struct ModelYuNetRt {
    gpu_buffers: Vec<CudaBuffer>,
    context: TrtContext,
    stream: CudaStream,
    engine: TrtEngine,

    input_width: u32,
    input_height: u32,

    tensor_buf: ndarray::Array4<f32>,
    /// Host buffers for all output bindings (indexed by binding index).
    output_host_bufs: Vec<Vec<f32>>,

    bindings: Vec<BindingInfo>,
    input_binding_idx: usize,
    stride_bindings: Vec<StrideBindings>,
}

impl ModelYuNetRt {
    /// Creates a new YuNet model from a TensorRT engine file.
    ///
    /// Input dimensions and output binding layout are read from the engine.
    ///
    /// # Arguments
    /// * `engine_path` - Path to the pre-built `.engine` file
    pub fn new_from_file(engine_path: &str) -> Result<Self, TrtModelError> {
        let engine = TrtEngine::from_file(engine_path)?;
        let bindings = engine.bindings();

        let input_binding_idx = bindings.iter()
            .position(|b| b.is_input)
            .ok_or_else(|| TrtModelError::Trt("No input binding found".into()))?;

        let input_dims = &bindings[input_binding_idx].dims;
        if input_dims.len() != 4 || input_dims[0] != 1 || input_dims[1] != 3 {
            return Err(TrtModelError::InvalidOutputShape(
                format!("Expected input [1, 3, H, W], got {:?}", input_dims),
            ));
        }
        let input_width = input_dims[3] as u32;
        let input_height = input_dims[2] as u32;

        // Find output bindings for each stride
        let mut stride_bindings = Vec::new();
        for &s in &STRIDES {
            let cls_name = format!("cls_{}", s);
            let obj_name = format!("obj_{}", s);
            let bbox_name = format!("bbox_{}", s);
            let kps_name = format!("kps_{}", s);

            let find = |name: &str| -> Result<usize, TrtModelError> {
                bindings.iter()
                    .position(|b| !b.is_input && b.name == name)
                    .ok_or_else(|| TrtModelError::Trt(format!("Output binding '{}' not found", name)))
            };

            stride_bindings.push(StrideBindings {
                stride: s,
                cls_idx: find(&cls_name)?,
                obj_idx: find(&obj_name)?,
                bbox_idx: find(&bbox_name)?,
                kps_idx: find(&kps_name)?,
            });
        }

        // Allocate GPU buffers for all bindings
        let mut gpu_buffers = Vec::with_capacity(bindings.len());
        for binding in &bindings {
            gpu_buffers.push(CudaBuffer::new(binding.byte_size)?);
        }

        // Allocate host buffers for output bindings only
        let mut output_host_bufs = Vec::with_capacity(bindings.len());
        for binding in &bindings {
            if binding.is_input {
                output_host_bufs.push(Vec::new());
            } else {
                let num_floats = binding.byte_size / 4;
                output_host_bufs.push(vec![0.0f32; num_floats]);
            }
        }

        let context = engine.create_context()?;
        let stream = CudaStream::new()?;

        Ok(Self {
            engine,
            context,
            stream,
            input_width,
            input_height,
            tensor_buf: ndarray::Array4::zeros((1, 3, input_height as usize, input_width as usize)),
            output_host_bufs,
            gpu_buffers,
            bindings,
            input_binding_idx,
            stride_bindings,
        })
    }

    /// Returns the input size (width, height).
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Runs face detection on an image.
    ///
    /// # Arguments
    /// * `image` - Input image buffer (RGB)
    /// * `conf_threshold` - Confidence threshold (0.0 - 1.0)
    /// * `nms_threshold` - NMS IoU threshold (0.0 - 1.0)
    pub fn forward(
        &mut self,
        image: &ImageBuffer,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, TrtModelError> {
        // 1. Preprocess: letterbox resize + RGB=>BGR, unnormalized
        let meta = PreprocessMeta::Letterbox(preprocess_into_bgr_letterbox(image, &mut self.tensor_buf));

        // 2. H2D: copy input tensor to GPU
        let input_bytes = unsafe {
            std::slice::from_raw_parts(
                self.tensor_buf.as_ptr() as *const u8,
                self.bindings[self.input_binding_idx].byte_size,
            )
        };
        self.gpu_buffers[self.input_binding_idx].copy_from_host(input_bytes, &self.stream)?;

        // 3. Build binding pointers
        let mut binding_ptrs: Vec<*mut c_void> = self.gpu_buffers
            .iter()
            .map(|buf| buf.as_ptr())
            .collect();

        // 4. Inference
        self.context.enqueue(&mut binding_ptrs, &self.stream)?;

        // 5. D2H: copy all output bindings
        for (i, binding) in self.bindings.iter().enumerate() {
            if binding.is_input {
                continue;
            }
            let host_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    self.output_host_bufs[i].as_mut_ptr() as *mut u8,
                    binding.byte_size,
                )
            };
            self.gpu_buffers[i].copy_to_host(host_bytes, &self.stream)?;
        }

        // 6. Synchronize
        self.stream.synchronize()?;

        // 7. Decode per stride
        let iw = self.input_width as f32;
        let ih = self.input_height as f32;

        let mut detections = Vec::new();

        for sb in &self.stride_bindings {
            let feat_w = (iw / sb.stride as f32).ceil() as usize;
            let feat_h = (ih / sb.stride as f32).ceil() as usize;

            decode_yunet_stride(
                &self.output_host_bufs[sb.cls_idx],
                &self.output_host_bufs[sb.obj_idx],
                &self.output_host_bufs[sb.bbox_idx],
                &self.output_host_bufs[sb.kps_idx],
                sb.stride, feat_w, feat_h,
                &meta,
                conf_threshold,
                &mut detections,
            );
        }

        Ok(nms_faces(&detections, nms_threshold))
    }
}

impl FaceDetector for ModelYuNetRt {
    type Input = ImageBuffer;
    type Error = TrtModelError;

    fn detect_faces(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, Self::Error> {
        self.forward(input, conf_threshold, nms_threshold)
    }
}
