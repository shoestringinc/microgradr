#![allow(unused)]

// This is like Numpy equivalent - just wrappers using ndarray crate.

use ndarray::prelude::*;
use ndarray::Array;

pub fn arange(start: f64, stop: f64, step: f64) -> Array1<f64> {
    Array::range(start, stop, step)
}

pub fn max_array1(xs: &Array1<f64>) -> f64 {
    xs.iter().cloned().fold(f64::MIN, |a, b| a.max(b))
}

pub fn min_array1(xs: &Array1<f64>) -> f64 {
    xs.iter().cloned().fold(f64::MAX, |a, b| a.min(b))
}
