use opencv::{
    dnn::DNN_BACKEND_OPENCV,
    dnn::DNN_BACKEND_INFERENCE_ENGINE,
    dnn::DNN_BACKEND_HALIDE,
    dnn::DNN_BACKEND_CUDA,
    dnn::DNN_TARGET_CPU,
    dnn::DNN_TARGET_OPENCL,
    dnn::DNN_TARGET_OPENCL_FP16,
    dnn::DNN_TARGET_MYRIAD,
    dnn::DNN_TARGET_FPGA,
    dnn::DNN_TARGET_CUDA,
    dnn::DNN_TARGET_CUDA_FP16,
    dnn::DNN_TARGET_HDDL,
};
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use super::model_format::{
    ModelFormat,
    ModelVersion
};
use lazy_static::lazy_static;

// Utilize lazy_static macro to create compatibility table for OpenCV's DNN backends and targets valid combinations
lazy_static!{
    pub static ref BACKEND_TARGET_VALID: HashMap<i32, HashSet<i32>> = vec![
        (DNN_BACKEND_OPENCV, HashSet::from_iter(vec![DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16])),
        (DNN_BACKEND_INFERENCE_ENGINE, HashSet::from_iter(vec![DNN_TARGET_CPU, DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16, DNN_TARGET_MYRIAD, DNN_TARGET_FPGA, DNN_TARGET_HDDL])),
        (DNN_BACKEND_HALIDE, HashSet::from_iter(vec![DNN_TARGET_CPU, DNN_TARGET_OPENCL])),
        (DNN_BACKEND_CUDA, HashSet::from_iter(vec![DNN_TARGET_CUDA, DNN_TARGET_CUDA_FP16])),
    ].into_iter().collect();
}

lazy_static!{
    pub static ref FORMAT_VERSION_VALID: HashMap<ModelFormat, HashSet<ModelVersion>> = vec![
        (ModelFormat::Darknet, HashSet::from_iter(vec![ModelVersion::V3, ModelVersion::V4, ModelVersion::V7])),
        (ModelFormat::ONNX, HashSet::from_iter(vec![ModelVersion::V4, ModelVersion::V8])),
    ].into_iter().collect();
}


/// Implementation of minMaxLoc in OpenCV (see the ref. https://docs.opencv.org/4.8.0/d2/de8/group__core__array.html#gab473bf2eb6d14ff97e89b355dac20707) basically.
/// Returns min, max and its indices in array.
/// Notice that it is for &[&T] rather than &[T]
/// Basic usage:
/// ```
/// use od_opencv::utils::min_max_loc_partial;
/// let x = vec![&3.0, &4.0, &1.0, &2.0, &9.0, &0.5, &8.0];
/// let (min, max, min_loc, max_loc) = min_max_loc_partial(&x).unwrap();
/// println!("Min: {} at pos {}. Max: {} at pos {}", min, max, min_loc, max_loc);
/// ```
/// 
pub fn min_max_loc_partial<T: PartialOrd + Copy>(data: &[&T]) -> Option<(T, T, usize, usize)> {
    if data.is_empty() {
        return None;
    }

    let mut min_val = data[0];
    let mut min_loc = 0;
    let mut max_val = data[0];
    let mut max_loc = 0;

    for (i, &val) in data.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_loc = i;
        } else if val > max_val {
            max_val = val;
            max_loc = i;
        }
    }

    Some((*min_val, *max_val, min_loc, max_loc))
}

mod tests {
    #[test]
    fn test_min_max_loc_partial() {
        let x = vec![&3.0, &4.0, &1.0, &2.0, &9.0, &0.5, &8.0];
        let (min, max, min_loc, max_loc) = crate::utils::min_max_loc_partial(&x).unwrap();
        assert_eq!(0.5, min);
        assert_eq!(5, min_loc);
        assert_eq!(9.0, max);
        assert_eq!(4, max_loc);
    }
}