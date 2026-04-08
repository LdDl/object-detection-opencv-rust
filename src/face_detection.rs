//! Face detection types and postprocessing.
//!
//! This module provides the `FaceDetection` struct, `FaceDetector` trait,
//! NMS for face detections, and shared YuNet decoding logic.

/// Strides used by YuNet model.
#[cfg(any(feature = "ort-backend", feature = "rknn-backend", feature = "tensorrt-backend"))]
pub const STRIDES: [u32; 3] = [8, 16, 32];

/// A single face detection with bounding box, landmarks, and confidence.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// Top-left X coordinate (in original image pixels)
    pub x: f32,
    /// Top-left Y coordinate (in original image pixels)
    pub y: f32,
    /// Bounding box width (in original image pixels)
    pub width: f32,
    /// Bounding box height (in original image pixels)
    pub height: f32,
    /// 5 facial landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
    /// Each landmark is [x, y] in original image pixels.
    pub landmarks: [[f32; 2]; 5],
    /// Detection confidence score (0.0 - 1.0)
    pub confidence: f32,
}

impl FaceDetection {
    /// Returns the area of the bounding box.
    #[inline]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// A trait for face detection models.
///
/// This trait provides a backend-agnostic interface for running face detection.
///
/// # Type Parameters
/// * `Input` - The input image type (e.g., `ImageBuffer`)
/// * `Error` - The error type for this backend
pub trait FaceDetector {
    /// The input image type for this detector.
    type Input;
    /// The error type for this detector.
    type Error;

    /// Runs face detection on the input image.
    ///
    /// # Arguments
    /// * `input` - The input image
    /// * `conf_threshold` - Confidence threshold for filtering detections (0.0 to 1.0)
    /// * `nms_threshold` - Non-maximum suppression IoU threshold (0.0 to 1.0)
    ///
    /// # Returns
    /// A vector of face detections.
    fn detect_faces(
        &mut self,
        input: &Self::Input,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<FaceDetection>, Self::Error>;
}

/// Performs Non-Maximum Suppression on face detections.
///
/// # Arguments
/// * `detections` - Array of face detections to filter
/// * `iou_threshold` - IoU threshold for considering boxes as overlapping (0.0 - 1.0)
///
/// # Returns
/// Filtered set of face detections
pub fn nms_faces(detections: &[FaceDetection], iou_threshold: f32) -> Vec<FaceDetection> {
    if detections.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<_> = detections.iter().enumerate().collect();
    sorted.sort_by(|a, b| {
        b.1.confidence
            .partial_cmp(&a.1.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for (orig_idx, detection) in sorted.iter() {
        if suppressed[*orig_idx] {
            continue;
        }

        keep.push((*detection).clone());

        for (other_orig_idx, other) in sorted.iter() {
            if suppressed[*other_orig_idx] || orig_idx == other_orig_idx {
                continue;
            }

            if iou_face(detection, other) > iou_threshold {
                suppressed[*other_orig_idx] = true;
            }
        }
    }

    keep
}

/// Computes IoU between two face detections.
#[inline]
fn iou_face(a: &FaceDetection, b: &FaceDetection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let union = a.area() + b.area() - intersection;

    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}

#[cfg(any(feature = "ort-backend", feature = "rknn-backend", feature = "tensorrt-backend"))]
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decodes YuNet outputs for a single stride into face detections.
///
/// All four output slices are flat, row-major: `[1, N, channels]` flattened to `[N * channels]`.
/// Supports both stretch and letterbox preprocessing via `PreprocessMeta`.
///
/// # Arguments
/// * `cls` - Classification logits, shape `[N, 1]` flattened
/// * `obj` - Objectness logits, shape `[N, 1]` flattened
/// * `bbox` - Bbox offsets, shape `[N, 4]` flattened
/// * `kps` - Landmark offsets, shape `[N, 10]` flattened
/// * `stride` - Feature map stride (8, 16, or 32)
/// * `feat_w` - Feature map width
/// * `feat_h` - Feature map height
/// * `meta` - Preprocessing metadata for coordinate inverse transform
/// * `conf_threshold` - Minimum confidence to keep
/// * `out` - Output vector to append detections to
#[cfg(any(feature = "ort-backend", feature = "rknn-backend", feature = "tensorrt-backend"))]
pub fn decode_yunet_stride(
    cls: &[f32],
    obj: &[f32],
    bbox: &[f32],
    kps: &[f32],
    stride: u32,
    feat_w: usize,
    feat_h: usize,
    meta: &crate::preprocessing::PreprocessMeta,
    conf_threshold: f32,
    out: &mut Vec<FaceDetection>,
) {
    let n = feat_w * feat_h;
    let s = stride as f32;

    for idx in 0..n {
        let col = idx % feat_w;
        let row = idx / feat_w;

        let anchor_cx = (col as f32 + 0.5) * s;
        let anchor_cy = (row as f32 + 0.5) * s;

        let cls_score = sigmoid(cls[idx]);
        let obj_score = sigmoid(obj[idx]).clamp(0.0, 1.0);
        let score = (cls_score * obj_score).sqrt();

        if score < conf_threshold {
            continue;
        }

        let cx = anchor_cx + bbox[idx * 4] * s;
        let cy = anchor_cy + bbox[idx * 4 + 1] * s;
        let w = bbox[idx * 4 + 2].exp() * s;
        let h = bbox[idx * 4 + 3].exp() * s;

        let (det_x, det_y, det_w, det_h) = meta.inverse_transform(
            cx - w / 2.0, cy - h / 2.0, w, h,
        );

        let mut landmarks = [[0.0f32; 2]; 5];
        for k in 0..5 {
            let lx = anchor_cx + kps[idx * 10 + 2 * k] * s;
            let ly = anchor_cy + kps[idx * 10 + 2 * k + 1] * s;
            let (ox, oy, _, _) = meta.inverse_transform(lx, ly, 0.0, 0.0);
            landmarks[k][0] = ox;
            landmarks[k][1] = oy;
        }

        out.push(FaceDetection {
            x: det_x,
            y: det_y,
            width: det_w,
            height: det_h,
            landmarks,
            confidence: score,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_faces_no_overlap() {
        let detections = vec![
            FaceDetection {
                x: 0.0, y: 0.0, width: 10.0, height: 10.0,
                landmarks: [[0.0; 2]; 5], confidence: 0.9,
            },
            FaceDetection {
                x: 100.0, y: 100.0, width: 10.0, height: 10.0,
                landmarks: [[0.0; 2]; 5], confidence: 0.8,
            },
        ];
        let result = nms_faces(&detections, 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_nms_faces_full_overlap() {
        let detections = vec![
            FaceDetection {
                x: 0.0, y: 0.0, width: 10.0, height: 10.0,
                landmarks: [[0.0; 2]; 5], confidence: 0.9,
            },
            FaceDetection {
                x: 0.0, y: 0.0, width: 10.0, height: 10.0,
                landmarks: [[0.0; 2]; 5], confidence: 0.8,
            },
        ];
        let result = nms_faces(&detections, 0.5);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < 0.001);
    }

    #[test]
    #[cfg(any(feature = "ort-backend", feature = "rknn-backend", feature = "tensorrt-backend"))]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    #[cfg(any(feature = "ort-backend", feature = "rknn-backend", feature = "tensorrt-backend"))]
    fn test_decode_yunet_stride_basic() {
        use crate::preprocessing::{PreprocessMeta, StretchMeta};

        // 2x2 grid, stride 8
        let cls = vec![5.0, -5.0, -5.0, -5.0]; // only first cell is confident
        let obj = vec![5.0, -5.0, -5.0, -5.0];
        let bbox = vec![
            0.0, 0.0, 1.0, 1.0, // cell 0
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let kps = vec![0.0; 40]; // 4 cells * 10

        let meta = PreprocessMeta::Stretch(StretchMeta {
            scale_x: 1.0,
            scale_y: 1.0,
            original_width: 16,
            original_height: 16,
        });

        let mut out = Vec::new();
        decode_yunet_stride(
            &cls, &obj, &bbox, &kps,
            8, 2, 2,
            &meta,
            0.5,
            &mut out,
        );

        assert_eq!(out.len(), 1);
        assert!(out[0].confidence > 0.9);
    }
}
