//! Bounding box type for object detection results.
//!
//! This module provides a backend-agnostic bounding box type that can be used
//! with any inference backend. When the `opencv-backend` feature is enabled,
//! It provides seamless conversion to/from `opencv::core::Rect`.

/// A bounding box representing a detected object.
///
/// Coordinates are in pixels, with (x, y) being the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BBox {
    /// X coordinate of the top-left corner
    pub x: i32,
    /// Y coordinate of the top-left corner
    pub y: i32,
    /// Width of the bounding box
    pub width: i32,
    /// Height of the bounding box
    pub height: i32,
}

impl BBox {
    /// Creates a new bounding box.
    #[inline]
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    /// Creates a bounding box from center coordinates.
    ///
    /// # Arguments
    /// * `cx` - X coordinate of the center
    /// * `cy` - Y coordinate of the center
    /// * `width` - Width of the bounding box
    /// * `height` - Height of the bounding box
    #[inline]
    pub fn from_center(cx: f32, cy: f32, width: f32, height: f32) -> Self {
        Self {
            x: (cx - width / 2.0).round() as i32,
            y: (cy - height / 2.0).round() as i32,
            width: width.round() as i32,
            height: height.round() as i32,
        }
    }

    /// Returns the area of the bounding box.
    #[inline]
    pub fn area(&self) -> i32 {
        self.width * self.height
    }

    /// Returns the center coordinates of the bounding box.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (
            self.x as f32 + self.width as f32 / 2.0,
            self.y as f32 + self.height as f32 / 2.0,
        )
    }

    /// Calculates the Intersection over Union (IoU) with another bounding box.
    ///
    /// IoU is a measure of overlap between two bounding boxes, ranging from 0 to 1.
    pub fn iou(&self, other: &BBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        if union <= 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }

    /// Clamps the bounding box to fit within image boundaries.
    ///
    /// # Arguments
    /// * `img_width` - Width of the image
    /// * `img_height` - Height of the image
    pub fn clamp(&self, img_width: i32, img_height: i32) -> Self {
        let x = self.x.max(0);
        let y = self.y.max(0);
        let width = (self.width).min(img_width - x);
        let height = (self.height).min(img_height - y);

        Self { x, y, width, height }
    }
}

// OpenCV conversions - available with opencv-backend or ort-opencv-compat feature
#[cfg(any(feature = "opencv-backend", feature = "ort-opencv-compat"))]
mod opencv_impl {
    use super::BBox;
    use opencv::core::Rect;

    impl From<BBox> for Rect {
        fn from(bbox: BBox) -> Self {
            Rect::new(bbox.x, bbox.y, bbox.width, bbox.height)
        }
    }

    impl From<Rect> for BBox {
        fn from(rect: Rect) -> Self {
            BBox::new(rect.x, rect.y, rect.width, rect.height)
        }
    }

    impl From<&Rect> for BBox {
        fn from(rect: &Rect) -> Self {
            BBox::new(rect.x, rect.y, rect.width, rect.height)
        }
    }

    impl From<&BBox> for Rect {
        fn from(bbox: &BBox) -> Self {
            Rect::new(bbox.x, bbox.y, bbox.width, bbox.height)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_new() {
        let bbox = BBox::new(10, 20, 100, 200);
        assert_eq!(bbox.x, 10);
        assert_eq!(bbox.y, 20);
        assert_eq!(bbox.width, 100);
        assert_eq!(bbox.height, 200);
    }

    #[test]
    fn test_bbox_from_center() {
        let bbox = BBox::from_center(50.0, 100.0, 100.0, 200.0);
        assert_eq!(bbox.x, 0);
        assert_eq!(bbox.y, 0);
        assert_eq!(bbox.width, 100);
        assert_eq!(bbox.height, 200);
    }

    #[test]
    fn test_bbox_area() {
        let bbox = BBox::new(0, 0, 100, 200);
        assert_eq!(bbox.area(), 20000);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = BBox::new(0, 0, 10, 10);
        let b = BBox::new(20, 20, 10, 10);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_bbox_iou_full_overlap() {
        let a = BBox::new(0, 0, 10, 10);
        let b = BBox::new(0, 0, 10, 10);
        assert_eq!(a.iou(&b), 1.0);
    }

    #[test]
    fn test_bbox_iou_partial_overlap() {
        let a = BBox::new(0, 0, 10, 10);
        let b = BBox::new(5, 5, 10, 10);
        // Intersection: 5x5 = 25
        // Union: 100 + 100 - 25 = 175
        let iou = a.iou(&b);
        assert!((iou - 25.0 / 175.0).abs() < 0.001);
    }

    #[test]
    fn test_bbox_clamp() {
        let bbox = BBox::new(-10, -10, 100, 100);
        let clamped = bbox.clamp(50, 50);
        assert_eq!(clamped.x, 0);
        assert_eq!(clamped.y, 0);
        assert_eq!(clamped.width, 50);
        assert_eq!(clamped.height, 50);
    }
}

#[cfg(all(test, any(feature = "opencv-backend", feature = "ort-opencv-compat")))]
mod opencv_tests {
    use super::*;
    use opencv::core::Rect;

    #[test]
    fn test_bbox_to_rect() {
        let bbox = BBox::new(10, 20, 100, 200);
        let rect: Rect = bbox.into();
        assert_eq!(rect.x, 10);
        assert_eq!(rect.y, 20);
        assert_eq!(rect.width, 100);
        assert_eq!(rect.height, 200);
    }

    #[test]
    fn test_rect_to_bbox() {
        let rect = Rect::new(10, 20, 100, 200);
        let bbox: BBox = rect.into();
        assert_eq!(bbox.x, 10);
        assert_eq!(bbox.y, 20);
        assert_eq!(bbox.width, 100);
        assert_eq!(bbox.height, 200);
    }
}
