//! Preprocessing utilities for object detection.
//!
//! This module provides pure-Rust image preprocessing functions that work
//! without OpenCV. These functions prepare images for inference.

use ndarray::Array4;
use crate::image_buffer::ImageBuffer;

/// Metadata from letterbox preprocessing, needed to reverse the transformation.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxMeta {
    /// Scale factor applied to the image
    pub scale: f32,
    /// Padding added to the left side
    pub pad_left: i32,
    /// Padding added to the top side
    pub pad_top: i32,
    /// Original image width
    pub original_width: i32,
    /// Original image height
    pub original_height: i32,
}

impl LetterboxMeta {
    /// Transforms coordinates from model output space back to original image space.
    #[inline]
    pub fn inverse_transform(&self, x: f32, y: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        (
            (x - self.pad_left as f32) / self.scale,
            (y - self.pad_top as f32) / self.scale,
            w / self.scale,
            h / self.scale,
        )
    }
}

/// Metadata from stretch preprocessing.
#[derive(Debug, Clone, Copy)]
pub struct StretchMeta {
    /// Scale factor for X axis
    pub scale_x: f32,
    /// Scale factor for Y axis
    pub scale_y: f32,
    /// Original image width
    pub original_width: i32,
    /// Original image height
    pub original_height: i32,
}

impl StretchMeta {
    /// Transforms coordinates from model output space back to original image space.
    #[inline]
    pub fn inverse_transform(&self, x: f32, y: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        (
            x * self.scale_x,
            y * self.scale_y,
            w * self.scale_x,
            h * self.scale_y,
        )
    }
}

/// Union type for preprocessing metadata.
#[derive(Debug, Clone, Copy)]
pub enum PreprocessMeta {
    Letterbox(LetterboxMeta),
    Stretch(StretchMeta),
}

impl PreprocessMeta {
    /// Transforms coordinates from model output space back to original image space.
    #[inline]
    pub fn inverse_transform(&self, x: f32, y: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        match self {
            PreprocessMeta::Letterbox(meta) => meta.inverse_transform(x, y, w, h),
            PreprocessMeta::Stretch(meta) => meta.inverse_transform(x, y, w, h),
        }
    }
}

// Pure Rust preprocessing using the `image` crate
#[cfg(feature = "ort-backend")]
mod image_preprocessing {
    use super::*;
    use image::{imageops::FilterType, Rgb, RgbImage};

    /// Resizes an image to the target size by stretching (may distort aspect ratio).
    pub fn resize_stretch(
        img: &ImageBuffer,
        target_width: u32,
        target_height: u32,
    ) -> (ImageBuffer, StretchMeta) {
        let (orig_height, orig_width, _) = img.shape();

        let meta = StretchMeta {
            scale_x: orig_width as f32 / target_width as f32,
            scale_y: orig_height as f32 / target_height as f32,
            original_width: orig_width as i32,
            original_height: orig_height as i32,
        };

        let dyn_img = img.to_dynamic_image();
        let resized = dyn_img.resize_exact(target_width, target_height, FilterType::Triangle);
        let resized_buf = ImageBuffer::from_dynamic_image(resized);

        (resized_buf, meta)
    }

    /// Resizes an image while preserving aspect ratio and padding with gray.
    pub fn resize_letterbox(
        img: &ImageBuffer,
        target_width: u32,
        target_height: u32,
    ) -> (ImageBuffer, LetterboxMeta) {
        let (orig_height, orig_width, _) = img.shape();

        let scale = f32::min(
            target_width as f32 / orig_width as f32,
            target_height as f32 / orig_height as f32,
        );

        let new_width = (orig_width as f32 * scale).round() as u32;
        let new_height = (orig_height as f32 * scale).round() as u32;

        let pad_left = ((target_width - new_width) / 2) as i32;
        let pad_top = ((target_height - new_height) / 2) as i32;

        let meta = LetterboxMeta {
            scale,
            pad_left,
            pad_top,
            original_width: orig_width as i32,
            original_height: orig_height as i32,
        };

        let dyn_img = img.to_dynamic_image();
        let resized = dyn_img.resize_exact(new_width, new_height, FilterType::Triangle);

        // Create padded image with gray background (114, 114, 114)
        let mut padded = RgbImage::from_pixel(target_width, target_height, Rgb([114, 114, 114]));

        let resized_rgb = resized.to_rgb8();
        for y in 0..new_height {
            for x in 0..new_width {
                let pixel = resized_rgb.get_pixel(x, y);
                padded.put_pixel(x + pad_left as u32, y + pad_top as u32, *pixel);
            }
        }

        let result = ImageBuffer::from_rgb_image(padded);
        (result, meta)
    }

    /// Converts an ImageBuffer to a normalized float32 tensor in NCHW format.
    pub fn to_nchw_tensor(img: &ImageBuffer) -> Array4<f32> {
        let (height, width, channels) = img.shape();
        let data = img.as_array();

        let mut tensor = Array4::<f32>::zeros((1, channels, height, width));

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    tensor[[0, c, h, w]] = data[[h, w, c]] as f32 / 255.0;
                }
            }
        }

        tensor
    }

    /// Full preprocessing pipeline: resize + normalize.
    pub fn preprocess(
        img: &ImageBuffer,
        target_width: u32,
        target_height: u32,
        use_letterbox: bool,
    ) -> (Array4<f32>, PreprocessMeta) {
        let (resized, meta) = if use_letterbox {
            let (resized, meta) = resize_letterbox(img, target_width, target_height);
            (resized, PreprocessMeta::Letterbox(meta))
        } else {
            let (resized, meta) = resize_stretch(img, target_width, target_height);
            (resized, PreprocessMeta::Stretch(meta))
        };

        let tensor = to_nchw_tensor(&resized);
        (tensor, meta)
    }
}

#[cfg(feature = "ort-backend")]
pub use image_preprocessing::*;

#[cfg(test)]
#[cfg(feature = "ort-backend")]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_resize_stretch() {
        let data = Array3::from_elem((480, 640, 3), 128u8);
        let img = ImageBuffer::from_rgb(data);

        let (resized, meta) = resize_stretch(&img, 320, 320);

        assert_eq!(resized.width(), 320);
        assert_eq!(resized.height(), 320);
        assert!((meta.scale_x - 2.0).abs() < 0.01);
        assert!((meta.scale_y - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_resize_letterbox() {
        let data = Array3::from_elem((480, 640, 3), 128u8);
        let img = ImageBuffer::from_rgb(data);

        let (resized, meta) = resize_letterbox(&img, 640, 640);

        assert_eq!(resized.width(), 640);
        assert_eq!(resized.height(), 640);
        assert!(meta.scale > 0.0);
        assert!(meta.pad_top > 0 || meta.pad_left > 0);
    }

    #[test]
    fn test_to_nchw_tensor() {
        let mut data = Array3::zeros((2, 3, 3));
        data[[0, 0, 0]] = 255; // R=255 at (0,0)

        let img = ImageBuffer::from_rgb(data);
        let tensor = to_nchw_tensor(&img);

        assert_eq!(tensor.shape(), &[1, 3, 2, 3]);
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_letterbox_inverse_transform() {
        let meta = LetterboxMeta {
            scale: 0.5,
            pad_left: 10,
            pad_top: 20,
            original_width: 640,
            original_height: 480,
        };

        let (x, y, w, h) = meta.inverse_transform(110.0, 120.0, 50.0, 50.0);

        assert!((x - 200.0).abs() < 0.01);
        assert!((y - 200.0).abs() < 0.01);
        assert!((w - 100.0).abs() < 0.01);
        assert!((h - 100.0).abs() < 0.01);
    }
}
