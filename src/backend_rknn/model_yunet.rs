//! YuNet face detection model using RKNN NPU.
//!
//! Model: face_detection_yunet_2023mar (OpenCV Zoo)
//! Input: [1, H, W, 3] uint8, RGB (RKNN handles channel order internally)
//! Outputs: 12 tensors (4 per stride 8/16/32): cls, obj, bbox, kps
//!
//! RKNN may reorder or rename outputs during conversion. The constructor
//! identifies outputs by shape: for each stride, cls has 1 channel, obj has 1,
//! bbox has 4, and kps has 10.

use rknn_runtime::RknnModel;

use crate::face_detection::{FaceDetection, FaceDetector, STRIDES, decode_yunet_stride, nms_faces};
use crate::image_buffer::ImageBuffer;
use crate::preprocessing::{StretchMeta, PreprocessMeta};

use super::RknnModelError;

/// Per-stride output indices into the RKNN output list.
struct StrideOutputs {
    stride: u32,
    cls_idx: usize,
    obj_idx: usize,
    bbox_idx: usize,
    kps_idx: usize,
}

/// YuNet face detection model using RKNN NPU.
pub struct ModelYuNetRknn {
    model: RknnModel,
    input_width: u32,
    input_height: u32,
    resize_buf: Vec<u8>,
    stride_outputs: Vec<StrideOutputs>,
}

impl ModelYuNetRknn {
    /// Creates a new YuNet model from an RKNN file.
    ///
    /// Input size is read from the model automatically.
    /// Output indices are matched by shape (channel count per stride).
    ///
    /// # Arguments
    /// * `model_path` - Path to the `.rknn` model file
    pub fn new_from_file(model_path: &str) -> Result<Self, RknnModelError> {
        let model = RknnModel::load(model_path)?;
        Self::from_model(model)
    }

    /// Creates a new YuNet model with a custom library path.
    ///
    /// # Arguments
    /// * `model_path` - Path to the `.rknn` model file
    /// * `lib_path` - Path to `librknnmrt.so`
    pub fn new_with_lib(model_path: &str, lib_path: &str) -> Result<Self, RknnModelError> {
        let model = RknnModel::load_with_lib(model_path, lib_path)?;
        Self::from_model(model)
    }

    fn from_model(model: RknnModel) -> Result<Self, RknnModelError> {
        let input_shape = &model.input_attr().shape;
        let input_height = input_shape[1];
        let input_width = input_shape[2];

        let output_attrs = model.output_attrs();
        let num_outputs = output_attrs.len();
        if num_outputs != 12 {
            return Err(RknnModelError::InvalidOutputShape(
                format!("Expected 12 outputs for YuNet, got {}", num_outputs),
            ));
        }

        // Match outputs to strides by expected spatial dimensions.
        // For each stride s: feat_h = ceil(H/s), feat_w = ceil(W/s), N = feat_h * feat_w
        // cls: [1, N, 1], obj: [1, N, 1], bbox: [1, N, 4], kps: [1, N, 10]
        let ih = input_height as f32;
        let iw = input_width as f32;

        let mut stride_outputs = Vec::new();
        for &s in &STRIDES {
            let feat_h = (ih / s as f32).ceil() as u32;
            let feat_w = (iw / s as f32).ceil() as u32;
            let n = feat_h * feat_w;

            // Find outputs matching this stride's spatial count
            let mut cls_idx = None;
            let mut obj_idx = None;
            let mut bbox_idx = None;
            let mut kps_idx = None;

            for (i, attr) in output_attrs.iter().enumerate() {
                let shape = &attr.shape;
                // Flatten: total elements = product of dims
                let total: u32 = shape.iter().product();
                // Match by total element count and last-dim channel count
                let last_dim = *shape.last().unwrap_or(&0);

                if total == n * 1 && last_dim == 1 {
                    if cls_idx.is_none() {
                        cls_idx = Some(i);
                    } else if obj_idx.is_none() {
                        obj_idx = Some(i);
                    }
                } else if total == n * 4 && last_dim == 4 {
                    bbox_idx = Some(i);
                } else if total == n * 10 && last_dim == 10 {
                    kps_idx = Some(i);
                }
            }

            stride_outputs.push(StrideOutputs {
                stride: s,
                cls_idx: cls_idx.ok_or_else(|| RknnModelError::InvalidOutputShape(
                    format!("cls output not found for stride {}", s),
                ))?,
                obj_idx: obj_idx.ok_or_else(|| RknnModelError::InvalidOutputShape(
                    format!("obj output not found for stride {}", s),
                ))?,
                bbox_idx: bbox_idx.ok_or_else(|| RknnModelError::InvalidOutputShape(
                    format!("bbox output not found for stride {}", s),
                ))?,
                kps_idx: kps_idx.ok_or_else(|| RknnModelError::InvalidOutputShape(
                    format!("kps output not found for stride {}", s),
                ))?,
            });
        }

        Ok(Self {
            model,
            input_width,
            input_height,
            resize_buf: vec![0u8; input_width as usize * input_height as usize * 3],
            stride_outputs,
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
    ) -> Result<Vec<FaceDetection>, RknnModelError> {
        let (orig_h, orig_w, _) = image.shape();
        let dst_w = self.input_width as usize;
        let dst_h = self.input_height as usize;
        let already_correct_size = orig_w == dst_w && orig_h == dst_h;

        // Preprocess: resize into pre-allocated buffer
        let meta = if already_correct_size {
            PreprocessMeta::Stretch(StretchMeta {
                scale_x: 1.0,
                scale_y: 1.0,
                original_width: orig_w as i32,
                original_height: orig_h as i32,
            })
        } else {
            let src = image.as_slice().expect("ImageBuffer not contiguous");
            resize_nearest_rgb_into(src, orig_w, orig_h, &mut self.resize_buf, dst_w, dst_h);
            PreprocessMeta::Stretch(StretchMeta {
                scale_x: orig_w as f32 / dst_w as f32,
                scale_y: orig_h as f32 / dst_h as f32,
                original_width: orig_w as i32,
                original_height: orig_h as i32,
            })
        };

        let input_bytes = if already_correct_size {
            image.as_slice().expect("ImageBuffer not contiguous")
        } else {
            &self.resize_buf
        };
        self.model.run(input_bytes)?;

        let iw = self.input_width as f32;
        let ih = self.input_height as f32;

        let mut detections = Vec::new();

        for so in &self.stride_outputs {
            let feat_w = (iw / so.stride as f32).ceil() as usize;
            let feat_h = (ih / so.stride as f32).ceil() as usize;

            // Use dequantized f32 outputs (YuNet is tiny, no perf concern)
            let cls = self.model.output_f32(so.cls_idx)?;
            let obj = self.model.output_f32(so.obj_idx)?;
            let bbox = self.model.output_f32(so.bbox_idx)?;
            let kps = self.model.output_f32(so.kps_idx)?;

            decode_yunet_stride(
                &cls, &obj, &bbox, &kps,
                so.stride, feat_w, feat_h,
                &meta,
                conf_threshold,
                &mut detections,
            );
        }

        Ok(nms_faces(&detections, nms_threshold))
    }
}

impl FaceDetector for ModelYuNetRknn {
    type Input = ImageBuffer;
    type Error = RknnModelError;

    fn detect_faces(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, Self::Error> {
        self.forward(input, conf_threshold, nms_threshold)
    }
}

/// Nearest-neighbor stretch resize into a caller-provided buffer.
#[inline(never)]
fn resize_nearest_rgb_into(
    src: &[u8], src_w: usize, src_h: usize,
    dst: &mut [u8], dst_w: usize, dst_h: usize,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for y in 0..dst_h {
        let src_y = (y * src_h) / dst_h;
        let dst_row = y * dst_w * 3;
        let src_row = src_y * src_w * 3;

        for x in 0..dst_w {
            let src_x = (x * src_w) / dst_w;
            let si = src_row + src_x * 3;
            let di = dst_row + x * 3;
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr.add(si), dst_ptr.add(di), 3);
            }
        }
    }
}
