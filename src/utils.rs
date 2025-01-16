#![allow(unused)]

use rand::Rng;
use rand::distributions::Uniform;

pub fn rand_uniform_nums(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();

    let uniform_range = Uniform::new(-1., 1.);

    // 0..n is fine too :-). Just that I like this more.
    (1..=n).map(|_| rng.sample(uniform_range)).collect()
}

pub fn rand_uniform() -> f64 {
    rand_uniform_nums(1)[0]
}

pub fn f64_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < std::f64::EPSILON
}