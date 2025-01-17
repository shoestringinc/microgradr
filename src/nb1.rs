#![allow(unused)]

use crate::engine::*;
use crate::numr::*;
use crate::plot::*;
use crate::utils::*;

// Upto: 19:20 of Andrej Video
// https://www.youtube.com/watch?v=VMj-3S1tku0

fn f(x: f64) -> f64 {
    // 3x^2 -4x + 5
    3. * (x.powf(2.)) - 4. * x + 5.
}

fn ex1() {
    assert!(f64_eq(f(3.0), 20.));
    let xs = arange(-5., 5., 0.25);
    dbg!(&xs);
    let ys = xs.mapv(f);
    dbg!(&ys);
    plot_to_file(&xs, &ys, "plot.png");
}

fn ex2() {
    let h = 0.000_000_1;
    let x = 3.0;
    let fx = f(x);
    let fxh = f(x + h);
    dbg!(&fxh);
    let slope = (fxh - fx) / h;
    dbg!(&slope);

    // Verify:
    // Differentiate 3x^2 -4x  + 5
    // 6x - 4
    // for x = 3.0
    // 6*3 - 4 = 14
}

fn ex3() {
    let h = 0.000_000_1;
    let x = -3.0;
    let fx = f(x);
    let fxh = f(x + h);
    let slope = (fxh - fx) / h;
    dbg!(&slope);
}

fn ex4() {
    // At x = 2/3 slope is zero.
    let h = 0.000_000_1;
    let x = 2. / 3.;
    let fx = f(x);
    let fxh = f(x + h);
    let slope = (fxh - fx) / h; // effectively 0.
    dbg!(&slope);
}

fn ex5() {
    let a = 2.;
    let b = -3.;
    let c = 10.;
    let d = a * b + c;
    println!("{d}"); // 4
}

fn ex6() {
    let h = 0.000_000_1;

    // inputs
    let a = 2.;
    let b = -3.;
    let c = 10.;

    // dd/da
    {
        let d1 = a * b + c;

        let a = a + h;
        let d2 = a * b + c;

        println!("{d1}");
        println!("{d2}");
        println!("Slope dd/da: {}", (d2 - d1) / h); // should be -3

        // Using calculus rule:
        // dd/da = b ie.  -3
    }

    // dd/db
    {
        let d1 = a * b + c;

        let b = b + h;
        let d2 = a * b + c;

        println!("{d1}");
        println!("{d2}");
        println!("Slope dd/db: {}", (d2 - d1) / h); // should be 2

        // Using calculus rule:
        // dd/db = a ie.  2
    }

    // dd/dc
    {
        let d1 = a * b + c;

        let c = c + h;
        let d2 = a * b + c;

        println!("{d1}");
        println!("{d2}");
        println!("Slope dd/dc: {}", (d2 - d1) / h); // should be 2

        // Using calculus rule:
        // dd/dc = 1

    }
}

pub fn main() {
    println!("Notebook #1");
    ex6();
}

