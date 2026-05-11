//! Face alignment via affine warp using 5 facial landmarks.
//!
//! Given 5 landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
//! detected by YuNet, computes a similarity transform that maps them to
//! canonical ArcFace reference positions, then warps the source image
//! to produce a 112x112 aligned face crop.
//!
//! The algorithm used is a simplified Umeyama estimator for 2D similarity
//! transforms (rotation + uniform scale + translation).
//!
//! Reference: S. Umeyama, "Least-Squares Estimation of Transformation Parameters
//! Between Two Point Patterns", IEEE TPAMI, 1991.

use crate::image_buffer::ImageBuffer;
use ndarray::Array3;

/// Output size for ArcFace-aligned faces.
pub const ALIGNED_FACE_SIZE: u32 = 112;

/// ArcFace canonical reference landmarks for 112x112 crop.
///
/// Order: left_eye, right_eye, nose, left_mouth, right_mouth.
const ARCFACE_REF: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

/// Estimates a 2D similarity transform (Umeyama) from `src` to `dst` points.
///
/// Returns a 2x3 affine matrix `[[a, -b, tx], [b, a, ty]]` such that
/// `dst ≈ M * [src; 1]`.
fn estimate_similarity(src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> [[f32; 3]; 2] {
    // Compute centroids
    let (mut src_cx, mut src_cy) = (0.0f32, 0.0f32);
    let (mut dst_cx, mut dst_cy) = (0.0f32, 0.0f32);
    for i in 0..5 {
        src_cx += src[i].0;
        src_cy += src[i].1;
        dst_cx += dst[i].0;
        dst_cy += dst[i].1;
    }
    src_cx /= 5.0;
    src_cy /= 5.0;
    dst_cx /= 5.0;
    dst_cy /= 5.0;

    // Centered coordinates and covariance
    let mut s_xx = 0.0f32;
    let mut s_xy = 0.0f32;
    let mut s_yx = 0.0f32;
    let mut s_yy = 0.0f32;
    let mut src_var = 0.0f32;

    for i in 0..5 {
        let sx = src[i].0 - src_cx;
        let sy = src[i].1 - src_cy;
        let dx = dst[i].0 - dst_cx;
        let dy = dst[i].1 - dst_cy;

        s_xx += sx * dx;
        s_xy += sx * dy;
        s_yx += sy * dx;
        s_yy += sy * dy;
        src_var += sx * sx + sy * sy;
    }

    // For a similarity transform (rotation + uniform scale), the optimal
    // parameters from the covariance are:
    //   a = (s_xx + s_yy) / src_var
    //   b = (s_yx - s_xy) / src_var
    let a = (s_xx + s_yy) / src_var;
    let b = (s_yx - s_xy) / src_var;

    let tx = dst_cx - (a * src_cx - b * src_cy);
    let ty = dst_cy - (b * src_cx + a * src_cy);

    [[a, -b, tx], [b, a, ty]]
}

/// Aligns a face by warping the source image using 5 detected landmarks.
///
/// Produces a 112x112 RGB crop suitable for ArcFace embedding extraction.
///
/// # Arguments
/// * `image` - Source image (RGB, HWC)
/// * `landmarks` - 5 facial landmarks as `[[x, y]; 5]` in source image coordinates
///
/// # Returns
/// A 112x112 `ImageBuffer` containing the aligned face crop.
pub fn align_face(image: &ImageBuffer, landmarks: &[[f32; 2]; 5]) -> ImageBuffer {
    let src: [(f32, f32); 5] = [
        (landmarks[0][0], landmarks[0][1]),
        (landmarks[1][0], landmarks[1][1]),
        (landmarks[2][0], landmarks[2][1]),
        (landmarks[3][0], landmarks[3][1]),
        (landmarks[4][0], landmarks[4][1]),
    ];

    let m = estimate_similarity(&src, &ARCFACE_REF);
    warp_affine(image, &m, ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE)
}

/// Inverts a 2x3 affine matrix.
///
/// For `M = [[a, b, c], [d, e, f]]`, computes `M^{-1}` such that
/// `M^{-1} * M = I` (for the 2x2 part).
fn invert_affine(m: &[[f32; 3]; 2]) -> [[f32; 3]; 2] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];

    let det = a * e - b * d;
    let inv_a = e / det;
    let inv_b = -b / det;
    let inv_c = (b * f - e * c) / det;
    let inv_d = -d / det;
    let inv_e = a / det;
    let inv_f = (d * c - a * f) / det;

    [[inv_a, inv_b, inv_c], [inv_d, inv_e, inv_f]]
}

/// Applies a 2x3 affine warp to produce an output image of given size.
///
/// Uses backward mapping with bilinear interpolation, matching cv2.warpAffine behavior.
/// The forward transform `m` maps output -> input; this function inverts it internally
/// so that for each output pixel we sample the correct source location.
///
/// Out-of-bounds pixels are set to 0 (black).
fn warp_affine(image: &ImageBuffer, m: &[[f32; 3]; 2], out_w: u32, out_h: u32) -> ImageBuffer {
    let src = image.as_array();
    let (sh, sw, ch) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let mut dst = Array3::<u8>::zeros((out_h as usize, out_w as usize, ch));

    // Invert the forward transform to get dst->src mapping
    let inv = invert_affine(m);

    for y in 0..out_h as usize {
        for x in 0..out_w as usize {
            let src_x = inv[0][0] * x as f32 + inv[0][1] * y as f32 + inv[0][2];
            let src_y = inv[1][0] * x as f32 + inv[1][1] * y as f32 + inv[1][2];

            // Bounds check (bilinear needs x+1, y+1)
            if src_x >= 0.0 && src_x < (sw - 1) as f32
                && src_y >= 0.0 && src_y < (sh - 1) as f32
            {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let y1 = (y0 + 1).min(sh - 1);

                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                for c in 0..ch {
                    let v00 = src[[y0, x0, c]] as f32;
                    let v01 = src[[y0, x1, c]] as f32;
                    let v10 = src[[y1, x0, c]] as f32;
                    let v11 = src[[y1, x1, c]] as f32;

                    let v0 = v00 * (1.0 - dx) + v01 * dx;
                    let v1 = v10 * (1.0 - dx) + v11 * dx;
                    let v = v0 * (1.0 - dy) + v1 * dy;

                    dst[[y, x, c]] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    ImageBuffer::from_rgb(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_similarity_identity() {
        // Same src and dst should give identity-like transform
        let pts: [(f32, f32); 5] = [
            (10.0, 20.0),
            (30.0, 20.0),
            (20.0, 35.0),
            (12.0, 45.0),
            (28.0, 45.0),
        ];
        let m = estimate_similarity(&pts, &pts);
        // a ≈ 1, b ≈ 0, tx ≈ 0, ty ≈ 0
        assert!((m[0][0] - 1.0).abs() < 1e-4, "a = {}", m[0][0]);
        assert!(m[1][0].abs() < 1e-4, "b = {}", m[1][0]);
        assert!(m[0][2].abs() < 1e-3, "tx = {}", m[0][2]);
        assert!(m[1][2].abs() < 1e-3, "ty = {}", m[1][2]);
    }

    #[test]
    fn test_estimate_similarity_scale() {
        // dst = 2 * src
        let src: [(f32, f32); 5] = [
            (10.0, 20.0),
            (30.0, 20.0),
            (20.0, 35.0),
            (12.0, 45.0),
            (28.0, 45.0),
        ];
        let dst: [(f32, f32); 5] = [
            (20.0, 40.0),
            (60.0, 40.0),
            (40.0, 70.0),
            (24.0, 90.0),
            (56.0, 90.0),
        ];
        let m = estimate_similarity(&src, &dst);
        // scale ≈ 2, rotation ≈ 0
        assert!((m[0][0] - 2.0).abs() < 1e-3, "a = {}", m[0][0]);
        assert!(m[1][0].abs() < 1e-3, "b = {}", m[1][0]);
    }

    #[test]
    fn test_align_face_output_size() {
        let img = ImageBuffer::zeros(200, 300, 3);
        let landmarks = [
            [100.0, 80.0],
            [150.0, 80.0],
            [125.0, 110.0],
            [105.0, 135.0],
            [145.0, 135.0],
        ];
        let aligned = align_face(&img, &landmarks);
        assert_eq!(aligned.width(), 112);
        assert_eq!(aligned.height(), 112);
        assert_eq!(aligned.channels(), 3);
    }
}
