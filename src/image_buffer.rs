//! Image buffer type for storing image data.
//!
//! This module provides a backend-agnostic image buffer that can be used
//! with any inference backend. It uses `ndarray` internally for efficient
//! array operations.

use ndarray::{Array3, ArrayView3};

/// Color channel order for image data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    /// Red, Green, Blue (standard for most image libraries)
    RGB,
    /// Blue, Green, Red (used by OpenCV)
    BGR,
}

/// An image buffer storing pixel data.
///
/// The data is stored in HWC (Height, Width, Channels) format as `u8` values.
/// The channel order is always RGB internally.
#[derive(Debug, Clone)]
pub struct ImageBuffer {
    /// Image data in HWC format, RGB channel order
    data: Array3<u8>,
}

impl ImageBuffer {
    /// Creates a new ImageBuffer from raw ndarray data.
    ///
    /// # Arguments
    /// * `data` - Array in HWC format (Height, Width, Channels)
    /// * `channel_order` - The channel order of the input data
    ///
    /// If the input is BGR, it will be converted to RGB.
    pub fn from_ndarray(data: Array3<u8>, channel_order: ChannelOrder) -> Self {
        let data = match channel_order {
            ChannelOrder::RGB => data,
            ChannelOrder::BGR => Self::bgr_to_rgb(data),
        };
        Self { data }
    }

    /// Creates a new ImageBuffer from raw ndarray data that is already RGB.
    ///
    /// This is a convenience method that assumes RGB channel order.
    #[inline]
    pub fn from_rgb(data: Array3<u8>) -> Self {
        Self { data }
    }

    /// Creates a new ImageBuffer from BGR data.
    ///
    /// This is a convenience method for OpenCV-style BGR images.
    #[inline]
    pub fn from_bgr(data: Array3<u8>) -> Self {
        Self::from_ndarray(data, ChannelOrder::BGR)
    }

    /// Creates an empty (zeros filled) ImageBuffer with the specified dimensions.
    ///
    /// # Arguments
    /// * `height` - Image height
    /// * `width` - Image width
    /// * `channels` - Number of channels (typically 3 for RGB)
    pub fn zeros(height: usize, width: usize, channels: usize) -> Self {
        Self {
            data: Array3::zeros((height, width, channels)),
        }
    }

    /// Returns the height of the image.
    #[inline]
    pub fn height(&self) -> usize {
        self.data.shape()[0]
    }

    /// Returns the width of the image.
    #[inline]
    pub fn width(&self) -> usize {
        self.data.shape()[1]
    }

    /// Returns the number of channels.
    #[inline]
    pub fn channels(&self) -> usize {
        self.data.shape()[2]
    }

    /// Returns the dimensions as (height, width, channels).
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.height(), self.width(), self.channels())
    }

    /// Returns a view of the underlying data.
    #[inline]
    pub fn as_array(&self) -> ArrayView3<'_, u8> {
        self.data.view()
    }

    /// Returns a reference to the underlying Array3.
    #[inline]
    pub fn data(&self) -> &Array3<u8> {
        &self.data
    }

    /// Consumes the ImageBuffer and returns the underlying Array3.
    #[inline]
    pub fn into_array(self) -> Array3<u8> {
        self.data
    }

    /// Converts BGR array to RGB by swapping channels.
    fn bgr_to_rgb(mut data: Array3<u8>) -> Array3<u8> {
        // Swap B and R channels (indices 0 and 2)
        let shape = data.shape();
        let height = shape[0];
        let width = shape[1];

        for h in 0..height {
            for w in 0..width {
                let b = data[[h, w, 0]];
                let r = data[[h, w, 2]];
                data[[h, w, 0]] = r;
                data[[h, w, 2]] = b;
            }
        }
        data
    }

    /// Creates a view in BGR order (for OpenCV compatibility).
    /// Note: This creates a new array, not a view.
    pub fn to_bgr(&self) -> Array3<u8> {
        let mut bgr = self.data.clone();
        let shape = bgr.shape();
        let height = shape[0];
        let width = shape[1];

        for h in 0..height {
            for w in 0..width {
                let r = bgr[[h, w, 0]];
                let b = bgr[[h, w, 2]];
                bgr[[h, w, 0]] = b;
                bgr[[h, w, 2]] = r;
            }
        }
        bgr
    }
}

// OpenCV conversions - available with opencv-backend feature
#[cfg(feature = "opencv-backend")]
mod opencv_impl {
    use super::*;
    use opencv::core::{Mat, MatTraitConst};

    impl ImageBuffer {
        /// Creates an ImageBuffer from an OpenCV `Mat`.
        ///
        /// The Mat is expected to be in BGR format (OpenCV's default).
        /// This performs a copy and converts BGR to RGB.
        ///
        /// # Errors
        /// Returns an error if the Mat cannot be converted.
        pub fn from_mat(mat: &Mat) -> Result<Self, opencv::Error> {
            // Ensure the Mat is continuous for efficient conversion
            let mat = if mat.is_continuous() {
                mat.clone()
            } else {
                let mut continuous = Mat::default();
                mat.copy_to(&mut continuous)?;
                continuous
            };

            let rows = mat.rows() as usize;
            let cols = mat.cols() as usize;
            let channels = mat.channels() as usize;

            // Get raw data
            let data_ptr = mat.data();
            let total_bytes = rows * cols * channels;

            let raw_data: Vec<u8> = unsafe {
                std::slice::from_raw_parts(data_ptr, total_bytes).to_vec()
            };

            // Create ndarray from raw data
            let data = Array3::from_shape_vec((rows, cols, channels), raw_data)
                .map_err(|e| opencv::Error::new(
                    opencv::core::StsError,
                    format!("Failed to create ndarray: {}", e),
                ))?;

            // Convert BGR to RGB
            Ok(Self::from_ndarray(data, ChannelOrder::BGR))
        }

        /// Converts the ImageBuffer to an OpenCV `Mat`.
        ///
        /// The resulting Mat will be in BGR format.
        pub fn to_mat(&self) -> Result<Mat, opencv::Error> {
            let bgr_data = self.to_bgr();
            let (height, width, channels) = (
                bgr_data.shape()[0] as i32,
                bgr_data.shape()[1] as i32,
                bgr_data.shape()[2] as i32,
            );

            let raw: Vec<u8> = bgr_data.into_iter().collect();
            let mat = Mat::from_slice(&raw)?;
            mat.reshape_nd(channels, &[height, width])?.try_clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_buffer_zeros() {
        let buf = ImageBuffer::zeros(480, 640, 3);
        assert_eq!(buf.height(), 480);
        assert_eq!(buf.width(), 640);
        assert_eq!(buf.channels(), 3);
    }

    #[test]
    fn test_image_buffer_from_rgb() {
        let data = Array3::from_elem((100, 100, 3), 128u8);
        let buf = ImageBuffer::from_rgb(data);
        assert_eq!(buf.shape(), (100, 100, 3));
    }

    #[test]
    fn test_bgr_to_rgb_conversion() {
        // Create a BGR image where B=1, G=2, R=3
        let mut bgr_data = Array3::zeros((2, 2, 3));
        // B
        bgr_data[[0, 0, 0]] = 1;
        // G
        bgr_data[[0, 0, 1]] = 2;
        // R
        bgr_data[[0, 0, 2]] = 3;

        let buf = ImageBuffer::from_bgr(bgr_data);

        // After conversion, should be R=3, G=2, B=1
        // R
        assert_eq!(buf.data[[0, 0, 0]], 3);
        // G
        assert_eq!(buf.data[[0, 0, 1]], 2);
        // B
        assert_eq!(buf.data[[0, 0, 2]], 1);
    }
}
