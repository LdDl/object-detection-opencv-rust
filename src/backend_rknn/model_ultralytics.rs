//! Ultralytics YOLO models (v8, v9, v11) using RKNN NPU.
//!
//! Warning!!!: Aggressively optimized for embedded ARM for SPECIFIC setup: Cortex-A7 / RV1106:
//! - Pre-allocated resize buffer (zero allocation per frame)
//! - Custom nearest-neighbor resize on raw bytes (no image crate overhead)
//! - Precomputed NC1HWC2 channel offsets (no division in hot loop)
//! - i8-space confidence threshold (skip float math for rejected predictions)
//! - Lazy bbox dequantization (only for detections above threshold)
//! - Incremental offset arithmetic (addition, not multiplication)

use rknn_runtime::{RknnModel, Nc1hwc2Layout};

use crate::bbox::BBox;
use crate::image_buffer::ImageBuffer;
use crate::postprocess::{Detection, nms, filter_by_class, detections_to_vecs};
use crate::preprocessing::{LetterboxMeta, PreprocessMeta, StretchMeta};

/// Error type for RKNN model operations.
#[derive(Debug)]
pub enum RknnModelError {
    /// Error from RKNN runtime
    Rknn(rknn_runtime::Error),
    /// Invalid model output shape
    InvalidOutputShape(String),
}

impl std::fmt::Display for RknnModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RknnModelError::Rknn(e) => write!(f, "RKNN error: {}", e),
            RknnModelError::InvalidOutputShape(s) => write!(f, "Invalid output shape: {}", s),
        }
    }
}

impl std::error::Error for RknnModelError {}

impl From<rknn_runtime::Error> for RknnModelError {
    fn from(e: rknn_runtime::Error) -> Self {
        RknnModelError::Rknn(e)
    }
}

/// Ultralytics YOLO model (v8, v9, v11) using RKNN NPU.
///
/// Expects models converted with `onnx_to_rknn.py` which:
/// - Normalizes bbox coordinates to 0-1
/// - Applies sigmoid to class scores
/// - Reshapes to 4D to avoid the RV1106 zero-copy bug
///
/// Output layout and channel offsets are precomputed at load time
/// to eliminate all division from the per-frame hot path.
pub struct ModelUltralyticsRknn {
    model: RknnModel,
    input_width: u32,
    input_height: u32,
    class_filters: Vec<usize>,
    use_letterbox: bool,
    /// Pre-allocated buffer for resized input (avoids 307KB alloc per frame).
    resize_buf: Vec<u8>,
    /// NC1HWC2 output layout (precomputed from tensor attributes at load time).
    layout: Nc1hwc2Layout,
    /// Precomputed raw-data offset for each class channel relative to p_offset.
    /// Computed once at load time; eliminates ch/c2 and ch%c2 division from inner loop.
    class_raw_offsets: Vec<usize>,
}

impl ModelUltralyticsRknn {
    /// Creates a new model from an RKNN file.
    ///
    /// Input size is read from the model automatically (NHWC shape).
    /// Validates that the output is NC1HWC2 format and precomputes the
    /// channel layout for zero-division inference.
    ///
    /// # Arguments
    /// * `model_path` - Path to the `.rknn` model file
    /// * `num_classes` - Number of detection classes in the model
    /// * `class_filters` - List of class indices to detect (empty for all classes)
    ///
    /// # Example
    /// ```ignore
    /// let model = ModelUltralyticsRknn::new_from_file(
    ///     "yolov8n.rknn",
    ///     80,       // COCO classes
    ///     vec![],   // detect all classes
    /// )?;
    /// ```
    pub fn new_from_file(
        model_path: &str,
        num_classes: usize,
        class_filters: Vec<usize>,
    ) -> Result<Self, RknnModelError> {
        let model = RknnModel::load(model_path)?;
        Self::from_model(model, num_classes, class_filters)
    }

    /// Creates a new model with a custom library path.
    ///
    /// Use this when `librknnmrt.so` is not at the default path (`/usr/lib/librknnmrt.so`).
    ///
    /// # Arguments
    /// * `model_path` - Path to the `.rknn` model file
    /// * `lib_path` - Path to `librknnmrt.so`
    /// * `num_classes` - Number of detection classes in the model
    /// * `class_filters` - List of class indices to detect (empty for all classes)
    ///
    /// # Example
    /// ```ignore
    /// let model = ModelUltralyticsRknn::new_with_lib(
    ///     "yolov8n.rknn",
    ///     "/opt/rknn/lib/librknnmrt.so",
    ///     80,
    ///     vec![],
    /// )?;
    /// ```
    pub fn new_with_lib(
        model_path: &str,
        lib_path: &str,
        num_classes: usize,
        class_filters: Vec<usize>,
    ) -> Result<Self, RknnModelError> {
        let model = RknnModel::load_with_lib(model_path, lib_path)?;
        Self::from_model(model, num_classes, class_filters)
    }

    /// Shared init: validate output format and precompute layout.
    fn from_model(
        model: RknnModel,
        num_classes: usize,
        class_filters: Vec<usize>,
    ) -> Result<Self, RknnModelError> {
        // Read input size from the model (NHWC: [1, H, W, 3]).
        let input_shape = &model.input_attr().shape;
        let input_height = input_shape[1];
        let input_width = input_shape[2];

        // Build NC1HWC2 layout from output tensor attributes.
        // Validates format + shape, precomputes stride/offset params.
        let layout = model.output_nc1hwc2_layout(0)?;

        if layout.c2() < 4 {
            return Err(RknnModelError::InvalidOutputShape(
                format!("NC1HWC2 c2={} < 4: bbox channels must fit in one block", layout.c2()),
            ));
        }

        // Precompute class channel offsets (channels 4..4+num_classes).
        let class_raw_offsets = layout.precompute_channel_offsets(4, num_classes);

        Ok(Self {
            model,
            input_width,
            input_height,
            class_filters,
            #[cfg(feature = "letterbox")]
            use_letterbox: true,
            #[cfg(not(feature = "letterbox"))]
            use_letterbox: false,
            resize_buf: vec![0u8; input_width as usize * input_height as usize * 3],
            layout,
            class_raw_offsets,
        })
    }

    /// Enables or disables letterbox preprocessing.
    ///
    /// Letterbox preserves aspect ratio by padding with gray (114, 114, 114).
    /// Default is `true` when the `letterbox` feature is enabled, `false` otherwise.
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
    /// * `image` - Input image buffer (RGB)
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
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), RknnModelError> {
        let (orig_h, orig_w, _) = image.shape();
        let dst_w = self.input_width as usize;
        let dst_h = self.input_height as usize;
        let already_correct_size = orig_w == dst_w && orig_h == dst_h;

        // Preprocess: resize into pre-allocated buffer or zero-copy passthrough
        let meta = if already_correct_size {
            PreprocessMeta::Stretch(StretchMeta {
                scale_x: 1.0,
                scale_y: 1.0,
                original_width: orig_w as i32,
                original_height: orig_h as i32,
            })
        } else {
            let src = image.as_slice().expect("ImageBuffer not contiguous");
            if self.use_letterbox {
                let lm = resize_letterbox_nearest_into(
                    src, orig_w, orig_h,
                    &mut self.resize_buf, dst_w, dst_h,
                );
                PreprocessMeta::Letterbox(lm)
            } else {
                resize_nearest_rgb_into(
                    src, orig_w, orig_h,
                    &mut self.resize_buf, dst_w, dst_h,
                );
                PreprocessMeta::Stretch(StretchMeta {
                    scale_x: orig_w as f32 / dst_w as f32,
                    scale_y: orig_h as f32 / dst_h as f32,
                    original_width: orig_w as i32,
                    original_height: orig_h as i32,
                })
            }
        };

        // NPU inference - zero-copy when size matches, pre-allocated buffer otherwise
        let input_bytes = if already_correct_size {
            image.as_slice().expect("ImageBuffer not contiguous")
        } else {
            &self.resize_buf
        };
        self.model.run(input_bytes)?;

        // Parse NC1HWC2 output directly with precomputed offsets
        let raw = self.model.output_raw(0)?;
        let detections = parse_nc1hwc2_direct(
            raw,
            &self.class_raw_offsets,
            &self.layout,
            conf_threshold,
            self.input_width as f32,
            self.input_height as f32,
            &meta,
        );

        let filtered = filter_by_class(&detections, &self.class_filters);
        let final_detections = nms(&filtered, nms_threshold);
        Ok(detections_to_vecs(final_detections))
    }
}

impl crate::ObjectDetector for ModelUltralyticsRknn {
    type Input = ImageBuffer;
    type Error = RknnModelError;

    fn detect(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), Self::Error> {
        self.forward(input, conf_threshold, nms_threshold)
    }
}

// ---------------------------------------------------------------------------
// Fast preprocessing: nearest-neighbor resize into pre-allocated buffer.
// Pure Rust on raw RGB bytes - no image crate, no intermediate ImageBuffer.
// ---------------------------------------------------------------------------

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

/// Nearest-neighbor letterbox resize into a caller-provided buffer.
/// Preserves aspect ratio and pads with gray (114, 114, 114).
#[inline(never)]
fn resize_letterbox_nearest_into(
    src: &[u8], src_w: usize, src_h: usize,
    dst: &mut [u8], dst_w: usize, dst_h: usize,
) -> LetterboxMeta {
    let scale = f32::min(dst_w as f32 / src_w as f32, dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as usize;
    let new_h = (src_h as f32 * scale).round() as usize;
    let pad_left = (dst_w - new_w) / 2;
    let pad_top = (dst_h - new_h) / 2;

    // Gray padding
    dst.fill(114);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for y in 0..new_h {
        let src_y = (y * src_h) / new_h;
        let dst_row = (y + pad_top) * dst_w * 3;
        let src_row = src_y * src_w * 3;

        for x in 0..new_w {
            let src_x = (x * src_w) / new_w;
            let si = src_row + src_x * 3;
            let di = dst_row + (x + pad_left) * 3;
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr.add(si), dst_ptr.add(di), 3);
            }
        }
    }

    LetterboxMeta {
        scale,
        pad_left: pad_left as i32,
        pad_top: pad_top as i32,
        original_width: src_w as i32,
        original_height: src_h as i32,
    }
}

// ---------------------------------------------------------------------------
// Direct NC1HWC2 parser with precomputed offsets.
//
// Hot loop: one table lookup + one i8 load + one comparison per class.
// No division, no float math until a detection passes the i8 threshold.
// ---------------------------------------------------------------------------

/// Parse NC1HWC2 output directly from raw i8 data.
///
/// `class_raw_offsets` maps each class index to its raw-data offset relative
/// to the prediction base. These are precomputed at model load time,
/// eliminating `ch/c2` and `ch%c2` division from the inner loop.
///
/// `layout` provides prediction stride, dequantization, and i8 threshold
/// computation - all precomputed in `rknn-runtime`.
#[inline(never)]
fn parse_nc1hwc2_direct(
    raw: &[i8],
    class_raw_offsets: &[usize],
    layout: &Nc1hwc2Layout,
    conf_threshold: f32,
    input_width: f32,
    input_height: f32,
    meta: &PreprocessMeta,
) -> Vec<Detection> {
    let num_classes = class_raw_offsets.len();
    let threshold_i8 = layout.threshold_i8(conf_threshold);
    let stride = layout.prediction_stride();

    let mut detections = Vec::new();

    // Incremental p_offset: addition instead of p * stride multiplication
    let mut p_offset = 0usize;

    for _p in 0..layout.num_predictions() {
        // --- Scan class scores with precomputed offsets (no division) ---
        let mut best_raw = i8::MIN;
        let mut best_cls = 0usize;

        for (c, &off) in class_raw_offsets[..num_classes].iter().enumerate() {
            let v = unsafe { *raw.get_unchecked(off + p_offset) };
            if v > best_raw {
                best_raw = v;
                best_cls = c;
            }
        }

        // i8-space threshold: rejects ~99% of predictions with zero float math
        if best_raw >= threshold_i8 {
            // Dequantize only the winning score
            let best_conf = layout.dequant(best_raw);

            // Bbox channels 0-3 are at p_offset+0..3 (always in block 0, c2 >= 4)
            let cx = layout.dequant(unsafe { *raw.get_unchecked(p_offset) }) * input_width;
            let cy = layout.dequant(unsafe { *raw.get_unchecked(p_offset + 1) }) * input_height;
            let bw = layout.dequant(unsafe { *raw.get_unchecked(p_offset + 2) }) * input_width;
            let bh = layout.dequant(unsafe { *raw.get_unchecked(p_offset + 3) }) * input_height;

            if bw > 0.0 && bh > 0.0 {
                let (x, y, w_out, h_out) = meta.inverse_transform(cx, cy, bw, bh);
                detections.push(Detection::new(
                    BBox::from_center(x, y, w_out, h_out),
                    best_cls,
                    best_conf,
                ));
            }
        }

        p_offset += stride;
    }

    detections
}
