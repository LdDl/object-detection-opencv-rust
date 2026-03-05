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
#[cfg(any(feature = "ort-backend", feature = "rknn-backend"))]
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

    /// Converts an ImageBuffer (RGB) to a normalized float32 tensor in NCHW format.
    pub fn to_nchw_tensor(img: &ImageBuffer) -> Array4<f32> {
        let (height, width, channels) = img.shape();
        let mut tensor = Array4::<f32>::zeros((1, channels, height, width));
        let inv_255 = 1.0f32 / 255.0;

        if let (Some(src), Some(dst)) = (img.as_slice(), tensor.as_slice_mut()) {
            let hw = height * width;
            // Iterate in HWC order (cache-friendly for source)
            for h in 0..height {
                let row_offset = h * width;
                for w in 0..width {
                    let src_idx = (row_offset + w) * channels;
                    let dst_base = row_offset + w;
                    for c in 0..channels {
                        dst[c * hw + dst_base] = src[src_idx + c] as f32 * inv_255;
                    }
                }
            }
        } else {
            let data = img.as_array();
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        tensor[[0, c, h, w]] = data[[h, w, c]] as f32 * inv_255;
                    }
                }
            }
        }

        tensor
    }

    /// Converts a BGR HWC u8 array to RGB NCHW f32 tensor in one pass.
    ///
    /// This fuses the BGR→RGB conversion with normalization, saving one full
    /// image copy compared to separate operations.
    ///
    /// # Arguments
    /// * `bgr` - Input array in HWC format with BGR channel order
    ///
    /// # Returns
    /// Normalized f32 tensor in NCHW format with RGB channel order
    pub fn bgr_hwc_to_rgb_nchw_tensor(bgr: &ndarray::ArrayView3<u8>) -> Array4<f32> {
        let (height, width, _channels) = bgr.dim();
        let mut tensor = Array4::<f32>::zeros((1, 3, height, width));
        let inv_255 = 1.0f32 / 255.0;

        if let (Some(src), Some(dst)) = (bgr.as_slice(), tensor.as_slice_mut()) {
            let hw = height * width;
            for h in 0..height {
                let row_offset = h * width;
                for w in 0..width {
                    let src_idx = (row_offset + w) * 3;
                    let dst_base = row_offset + w;
                    // BGR → RGB swap fused with normalization
                    dst[dst_base] = src[src_idx + 2] as f32 * inv_255;          // R
                    dst[hw + dst_base] = src[src_idx + 1] as f32 * inv_255;     // G
                    dst[2 * hw + dst_base] = src[src_idx] as f32 * inv_255;     // B
                }
            }
        } else {
            for h in 0..height {
                for w in 0..width {
                    tensor[[0, 0, h, w]] = bgr[[h, w, 2]] as f32 * inv_255;
                    tensor[[0, 1, h, w]] = bgr[[h, w, 1]] as f32 * inv_255;
                    tensor[[0, 2, h, w]] = bgr[[h, w, 0]] as f32 * inv_255;
                }
            }
        }

        tensor
    }

    /// Fused bilinear resize + normalize into a pre-allocated NCHW f32 tensor.
    ///
    /// The tensor must have shape `(1, channels, target_height, target_width)`.
    /// This avoids per-frame tensor allocation.
    pub fn preprocess_into(
        img: &ImageBuffer,
        tensor: &mut Array4<f32>,
        use_letterbox: bool,
    ) -> PreprocessMeta {
        let (orig_height, orig_width, channels) = img.shape();
        let shape = tensor.shape();
        let th = shape[2];
        let tw = shape[3];

        if let Some(src) = img.as_slice() {
            let dst = tensor.as_slice_mut().unwrap();
            let hw = th * tw;
            let inv_255 = 1.0f32 / 255.0;

            if use_letterbox {
                let scale = f32::min(
                    tw as f32 / orig_width as f32,
                    th as f32 / orig_height as f32,
                );
                let new_width = (orig_width as f32 * scale).round() as usize;
                let new_height = (orig_height as f32 * scale).round() as usize;
                let pad_left = (tw - new_width) / 2;
                let pad_top = (th - new_height) / 2;

                let pad_val = 114.0f32 * inv_255;
                for c in 0..channels {
                    for i in 0..hw {
                        dst[c * hw + i] = pad_val;
                    }
                }

                bilinear_into(
                    src, orig_width, orig_height, channels,
                    dst, tw, new_width, new_height,
                    pad_left, pad_top, inv_255,
                );

                PreprocessMeta::Letterbox(LetterboxMeta {
                    scale,
                    pad_left: pad_left as i32,
                    pad_top: pad_top as i32,
                    original_width: orig_width as i32,
                    original_height: orig_height as i32,
                })
            } else {
                bilinear_into(
                    src, orig_width, orig_height, channels,
                    dst, tw, tw, th,
                    0, 0, inv_255,
                );

                PreprocessMeta::Stretch(StretchMeta {
                    scale_x: orig_width as f32 / tw as f32,
                    scale_y: orig_height as f32 / th as f32,
                    original_width: orig_width as i32,
                    original_height: orig_height as i32,
                })
            }
        } else {
            // Fallback: non-contiguous data
            let (new_tensor, meta) = preprocess(img, tw as u32, th as u32, use_letterbox);
            tensor.assign(&new_tensor);
            meta
        }
    }

    /// Full preprocessing pipeline: resize + normalize (allocating version).
    ///
    /// Uses a fused resize+normalize path when possible (contiguous source data),
    /// performing bilinear interpolation directly into the NCHW f32 tensor
    /// without intermediate image allocations.
    pub fn preprocess(
        img: &ImageBuffer,
        target_width: u32,
        target_height: u32,
        use_letterbox: bool,
    ) -> (Array4<f32>, PreprocessMeta) {
        let (_orig_height, _orig_width, channels) = img.shape();
        let th = target_height as usize;
        let tw = target_width as usize;

        let mut tensor = Array4::<f32>::zeros((1, channels, th, tw));
        let meta = preprocess_into(img, &mut tensor, use_letterbox);
        (tensor, meta)
    }

    /// Core bilinear interpolation loop writing into NCHW f32 slice.
    #[inline(always)]
    fn bilinear_into(
        src: &[u8], orig_width: usize, orig_height: usize, channels: usize,
        dst: &mut [f32], dst_stride: usize,
        new_width: usize, new_height: usize,
        pad_left: usize, pad_top: usize,
        inv_255: f32,
    ) {
        let hw = dst.len() / channels;
        let sx_scale = orig_width as f32 / new_width as f32;
        let sy_scale = orig_height as f32 / new_height as f32;

        for oh in 0..new_height {
            let sy = (oh as f32 + 0.5) * sy_scale - 0.5;
            let sy0 = (sy.floor() as isize).max(0) as usize;
            let sy1 = (sy0 + 1).min(orig_height - 1);
            let fy = sy - sy.floor();

            let dst_y = oh + pad_top;

            for ow in 0..new_width {
                let sx = (ow as f32 + 0.5) * sx_scale - 0.5;
                let sx0 = (sx.floor() as isize).max(0) as usize;
                let sx1 = (sx0 + 1).min(orig_width - 1);
                let fx = sx - sx.floor();

                let dst_base = dst_y * dst_stride + ow + pad_left;

                let w00 = (1.0 - fx) * (1.0 - fy);
                let w01 = fx * (1.0 - fy);
                let w10 = (1.0 - fx) * fy;
                let w11 = fx * fy;

                let s00 = (sy0 * orig_width + sx0) * channels;
                let s01 = (sy0 * orig_width + sx1) * channels;
                let s10 = (sy1 * orig_width + sx0) * channels;
                let s11 = (sy1 * orig_width + sx1) * channels;

                for c in 0..channels {
                    let val = src[s00 + c] as f32 * w00
                        + src[s01 + c] as f32 * w01
                        + src[s10 + c] as f32 * w10
                        + src[s11 + c] as f32 * w11;
                    dst[c * hw + dst_base] = val * inv_255;
                }
            }
        }
    }
}

#[cfg(any(feature = "ort-backend", feature = "rknn-backend"))]
pub use image_preprocessing::*;

#[cfg(test)]
#[cfg(any(feature = "ort-backend", feature = "rknn-backend"))]
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
        // R=255 at (0,0)
        data[[0, 0, 0]] = 255;

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

    #[test]
    fn test_bgr_hwc_to_rgb_nchw_tensor() {
        // Create a 2x2 BGR image
        // Pixel (0,0): B=10, G=20, R=30
        // Pixel (0,1): B=40, G=50, R=60
        let mut bgr_data = Array3::<u8>::zeros((2, 2, 3));
        // B
        bgr_data[[0, 0, 0]] = 10;
        // G
        bgr_data[[0, 0, 1]] = 20;
        // R
        bgr_data[[0, 0, 2]] = 30;
        // B
        bgr_data[[0, 1, 0]] = 40;
        // G
        bgr_data[[0, 1, 1]] = 50;
        // R
        bgr_data[[0, 1, 2]] = 60;

        let tensor = bgr_hwc_to_rgb_nchw_tensor(&bgr_data.view());

        // Check shape: NCHW = (1, 3, 2, 2)
        assert_eq!(tensor.shape(), &[1, 3, 2, 2]);

        // Check pixel (0,0): should be R=30/255, G=20/255, B=10/255
        // R channel
        assert!((tensor[[0, 0, 0, 0]] - 30.0 / 255.0).abs() < 0.001);
        // G channel
        assert!((tensor[[0, 1, 0, 0]] - 20.0 / 255.0).abs() < 0.001);
        // B channel
        assert!((tensor[[0, 2, 0, 0]] - 10.0 / 255.0).abs() < 0.001);

        // Check pixel (0,1): should be R=60/255, G=50/255, B=40/255
        // R channel
        assert!((tensor[[0, 0, 0, 1]] - 60.0 / 255.0).abs() < 0.001);
        // G channel
        assert!((tensor[[0, 1, 0, 1]] - 50.0 / 255.0).abs() < 0.001);
        // B channel
        assert!((tensor[[0, 2, 0, 1]] - 40.0 / 255.0).abs() < 0.001);
    }
}
