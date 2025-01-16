#![allow(unused)]

use crate::engine::*;
use crate::graphviz::*;

// Upto: 53:05 of Andrej Video
// https://www.youtube.com/watch?v=VMj-3S1tku0


fn ex1() {
    let a = Var::new(2.0, "a");
    let b = Var::new(-3.0, "b");
    // dbg!(&a);
    // dbg!(&a + &b);

    let c = Var::new(10.0, "c");
    let d = &(&a * &b) + &c;
    d.set_label("d");
    println!("d is {}", d.data());
    println!("d label: {}", d.label());
    println!("d op {}", d.op());

    draw_dot(&d, "graph"); // view graph.png in Preview or whatever browser or other app.
}

fn ex2() {
    let a = Var::new(2.0, "a");
    let b = Var::new(-3.0, "b");
    let c = Var::new(10.0, "c");
    let e = &a * &b;
    e.set_label("e");
    let d = &e + &c;
    d.set_label("d");

    let f = Var::new(-2.0, "f");
    let l = &d * &f; // think of l as a loss function.
    l.set_label("l");
    l.set_grad(1.0);

    l.back_propagate();

    draw_dot(&l, "graph");
    // You can verify - just the way Andrej did using the lol function.
    // That is the one below.
}

fn lol() {
    // Verifying the gradients created by the back propagation manually.

    let a = Var::new(2.0, "a");
    let b = Var::new(-3.0, "b");
    let c = Var::new(10.0, "c");
    let f = Var::new(-2.0, "f");

    let h = 0.000_000_1;

    // Convenience closure for quick calc. Just a random symbolic name since
    // we are doing neural nets.
    let lossfn = |a: &Var, b: &Var, c: &Var, f: &Var| {
        let e = a * b;
        let d = &e + c;
        let l = &d * f; // think of l as a loss function.
        l
    };

    let slopefn = |l1: &Var, l2: &Var| {
        (l2.data() - l1.data()) / h
    };

    {
        // dl/da
        let l1 = lossfn(&a, &b, &c, &f);
        let a = Var::new(a.data() + h, "");
        let l2 = lossfn(&a, &b, &c, &f);
        println!("dl/da: {}", slopefn(&l1, &l2)); // -4
    }

    {
        // dl/db
        let l1 = lossfn(&a, &b, &c, &f);
        let b = Var::new(b.data() + h, "");
        let l2 = lossfn(&a, &b, &c, &f);
        println!("dl/db: {}", slopefn(&l1, &l2)); // 6
    }

    {
        // dl/dc
        let l1 = lossfn(&a, &b, &c, &f);
        let c = Var::new(c.data() + h, "");
        let l2 = lossfn(&a, &b, &c, &f);
        println!("dl/dc: {}", slopefn(&l1, &l2)); // 6
    }

    {
        // dl/df
        let l1 = lossfn(&a, &b, &c, &f);
        let f = Var::new(f.data() + h, "");
        let l2 = lossfn(&a, &b, &c, &f);
        println!("dl/df: {}", slopefn(&l1, &l2)); // 6

    }

    {
        // dl/dd

        let e = &a * &b;
        let d = &e + &c;
        let l1 = &d * &f; 

        let d = Var::new(d.data() + h, "");
        let l2 = &d * &f; 
        println!("dl/dd {}", slopefn(&l1, &l2));

    }

    {
        // dl/de

        let e = &a * &b;
        let d = &e + &c;
        let l1 = &d * &f; 

        let e = Var::new(e.data() + h, "");
        let d = &e + &c;
        let l2 = &d * &f; 
        println!("dl/de {}", slopefn(&l1, &l2));

    }

    // Output
    // dl/da: 5.999999999062311
    // dl/db: -3.9999999934536845
    // dl/dc: -1.999999987845058
    // dl/df: 4.000000002335469
    // dl/dd -2.0000000056086265
    // dl/de -2.0000000056086265


}

pub fn main() {
    println!("Notebook 2");
    lol();
}
