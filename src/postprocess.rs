//! Post-processing utilities for object detection.
//!
//! This module provides backend-agnostic post-processing functions including
//! Non-Maximum Suppression (NMS) and output parsing.

use crate::bbox::BBox;

/// A single detection before NMS filtering.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box
    pub bbox: BBox,
    /// Class index
    pub class_id: usize,
    /// Confidence score
    pub confidence: f32,
}

impl Detection {
    /// Creates a new detection.
    #[inline]
    pub fn new(bbox: BBox, class_id: usize, confidence: f32) -> Self {
        Self {
            bbox,
            class_id,
            confidence,
        }
    }
}

/// Performs Non-Maximum Suppression on an array of detections.
///
/// NMS removes overlapping boxes, keeping only the highest confidence detection
/// for each object.
///
/// # Arguments
/// * `detections` - array of detections to filter
/// * `iou_threshold` - IoU threshold for considering boxes as overlapping (0.0 - 1.0)
///
/// # Returns
/// Filtered set of detections
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Sort by confidence (descending)
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

        // Suppress all detections with high IoU
        for (other_orig_idx, other) in sorted.iter() {
            if suppressed[*other_orig_idx] || orig_idx == other_orig_idx {
                continue;
            }

            let iou = detection.bbox.iou(&other.bbox);
            if iou > iou_threshold {
                suppressed[*other_orig_idx] = true;
            }
        }
    }

    keep
}

/// Performs class-aware Non-Maximum Suppression.
///
/// This variant only compares boxes within the same class.
///
/// # Arguments
/// * `detections` - array of detections to filter
/// * `iou_threshold` - IoU threshold for considering boxes as overlapping
///
/// # Returns
/// Filtered set of detections
pub fn nms_class_aware(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return Vec::new();
    }

    // Group by class
    let max_class = detections.iter().map(|d| d.class_id).max().unwrap_or(0);
    let mut by_class: Vec<Vec<&Detection>> = vec![Vec::new(); max_class + 1];

    for detection in detections {
        by_class[detection.class_id].push(detection);
    }

    // Apply NMS per class
    let mut result = Vec::new();
    for class_detections in by_class {
        if class_detections.is_empty() {
            continue;
        }

        // Convert to owned for NMS
        let owned: Vec<Detection> = class_detections.into_iter().cloned().collect();
        let filtered = nms(&owned, iou_threshold);
        result.extend(filtered);
    }

    result
}

/// Filters detections by confidence threshold.
///
/// # Arguments
/// * `detections` - array of detections to filter
/// * `threshold` - Minimum confidence threshold
///
/// # Returns
/// Filtered set of detections
#[inline]
pub fn filter_by_confidence(detections: &[Detection], threshold: f32) -> Vec<Detection> {
    detections
        .iter()
        .filter(|d| d.confidence >= threshold)
        .cloned()
        .collect()
}

/// Filters detections by class.
///
/// # Arguments
/// * `detections` - array of detections to filter
/// * `class_filter` - array of class IDs to keep (empty means keep all)
///
/// # Returns
/// Filtered set of detections
#[inline]
pub fn filter_by_class(detections: &[Detection], class_filter: &[usize]) -> Vec<Detection> {
    if class_filter.is_empty() {
        return detections.to_vec();
    }

    detections
        .iter()
        .filter(|d| class_filter.contains(&d.class_id))
        .cloned()
        .collect()
}

/// Converts detections to the output format (Vec<BBox>, Vec<usize>, Vec<f32>).
///
/// This matches the existing API format.
pub fn detections_to_vecs(detections: Vec<Detection>) -> (Vec<BBox>, Vec<usize>, Vec<f32>) {
    let mut bboxes = Vec::with_capacity(detections.len());
    let mut class_ids = Vec::with_capacity(detections.len());
    let mut confidences = Vec::with_capacity(detections.len());

    for det in detections {
        bboxes.push(det.bbox);
        class_ids.push(det.class_id);
        confidences.push(det.confidence);
    }

    (bboxes, class_ids, confidences)
}

/// Finds the index and value of the maximum element in a slice.
///
/// # Arguments
/// * `values` - array of f32 values
///
/// # Returns
/// Option<(index, max_value)>
#[inline]
pub fn argmax(values: &[f32]) -> Option<(usize, f32)> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, &val)| (idx, val))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detection(x: i32, y: i32, w: i32, h: i32, class_id: usize, conf: f32) -> Detection {
        Detection::new(BBox::new(x, y, w, h), class_id, conf)
    }

    #[test]
    fn test_nms_no_overlap() {
        let detections = vec![
            make_detection(0, 0, 10, 10, 0, 0.9),
            make_detection(100, 100, 10, 10, 0, 0.8),
        ];

        let result = nms(&detections, 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_nms_full_overlap() {
        let detections = vec![
            make_detection(0, 0, 10, 10, 0, 0.9),
            make_detection(0, 0, 10, 10, 0, 0.8),
        ];

        let result = nms(&detections, 0.5);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_nms_partial_overlap() {
        let detections = vec![
            make_detection(0, 0, 10, 10, 0, 0.9),
            // 25% overlap
            make_detection(5, 5, 10, 10, 0, 0.8),
        ];

        // With high threshold (0.9), both should be kept
        let result_high = nms(&detections, 0.9);
        assert_eq!(result_high.len(), 2);

        // With low threshold (0.1), one should be suppressed
        let result_low = nms(&detections, 0.1);
        assert_eq!(result_low.len(), 1);
    }

    #[test]
    fn test_nms_class_aware() {
        let detections = vec![
            // class 0
            make_detection(0, 0, 10, 10, 0, 0.9),
            // class 1, same location
            make_detection(0, 0, 10, 10, 1, 0.8),
        ];

        // Class-aware NMS should keep both (different classes)
        let result = nms_class_aware(&detections, 0.5);
        assert_eq!(result.len(), 2);

        // Regular NMS would suppress one
        let result_regular = nms(&detections, 0.5);
        assert_eq!(result_regular.len(), 1);
    }

    #[test]
    fn test_filter_by_confidence() {
        let detections = vec![
            make_detection(0, 0, 10, 10, 0, 0.9),
            make_detection(0, 0, 10, 10, 0, 0.3),
            make_detection(0, 0, 10, 10, 0, 0.5),
        ];

        let result = filter_by_confidence(&detections, 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_filter_by_class() {
        let detections = vec![
            make_detection(0, 0, 10, 10, 0, 0.9),
            make_detection(0, 0, 10, 10, 1, 0.8),
            make_detection(0, 0, 10, 10, 2, 0.7),
        ];

        let result = filter_by_class(&detections, &[0, 2]);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|d| d.class_id == 0 || d.class_id == 2));
    }

    #[test]
    fn test_argmax() {
        let values = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let (idx, max) = argmax(&values).unwrap();
        assert_eq!(idx, 3);
        assert!((max - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_detections_to_vecs() {
        let detections = vec![
            make_detection(10, 20, 30, 40, 5, 0.9),
            make_detection(50, 60, 70, 80, 3, 0.8),
        ];

        let (bboxes, class_ids, confidences) = detections_to_vecs(detections);

        assert_eq!(bboxes.len(), 2);
        assert_eq!(class_ids, vec![5, 3]);
        assert!((confidences[0] - 0.9).abs() < 0.001);
    }
}
