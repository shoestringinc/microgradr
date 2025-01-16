#![allow(unused)]

// Upto: 1:39:37 of Andrej Video
// https://www.youtube.com/watch?v=VMj-3S1tku0

use std::ops::Neg;

use burn::nn::Relu;

use crate::numr::*;
use crate::plot::*;
use crate::engine::*;
use crate::graphviz::*;
use crate::nn::*;


fn ex1() {
    let xs = arange(-5., 5., 0.2);
    let ys = xs.mapv(f64::tanh); // tanh is a the 'squashing' function
    plot_to_file(&xs, &ys, "plot.png");
}

fn ex2() {
    let x1 = Var::new(2.0, "x1");
    let x2 = Var::new(0.0, "x2");
    let w1 = Var::new(-3.0, "w1");
    let w2 = Var::new(1.0, "w2");
    let b = Var::new(6.8812735870195432, "b");

    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;

    let x1w1x2w2 = &x1w1 + &x2w2;
    
    let n = &x1w1x2w2 + &b;
    n.set_label("n");


    let o = n.tanh();
    o.set_label("o");
    // dbg!(&o);
    // println!("o value: {}", o.data());
    // o.dump_child_labels();

    o.back_propagate();

    draw_dot(&o, "weights");

}

fn ex3() {
    let a = Var::new(3.0, "a");
    let b = &a + &a;
    b.back_propagate();
    draw_dot(&b, "a_repeat_check");
}

fn ex4() {
    let a = Var::new(-2.0, "a");
    let b = Var::new(3.0, "b");
    let d = &a * &b;
    d.set_label("d");
    let e = &a + &b;
    e.set_label("e");
    let f = &d * &e;
    f.set_label("f");
    f.back_propagate();
    draw_dot(&f, "weights");
}

fn ex5() {
    let a = Var::new(2.0, "a");
    let b = &a + 1.;
    b.set_label("b");
    // draw_dot(&b, "weights");
    let c = &a * 2.;
    c.set_label("c");
    draw_dot(&c, "weights");

}

fn ex6() {
    let a = Var::new(2.0, "a");
    let b = 2. * &a;
    let c = 1. + &b;
    draw_dot(&c, "weights");
}

fn ex7() {
    let x = 2.0_f64;
    println!("{}", x.exp()); // 7.38905609893065

    let a = Var::new(2.0, "a");
    let b = a.exp();
    draw_dot(&b, "weights");
}

fn ex8() {
    let a = Var::new(2.0, "a");
    let b = Var::new(4.0, "b");
    let c = &a / &b;
    c.set_label("c");
    draw_dot(&c, "weights");
}

fn ex9() {
    let a = Var::new(2.0, "a");
    let b = Var::new(5.0, "b");
    let c = &a - &b;
    c.set_label("c");
    draw_dot(&c, "weights");
}

fn ex10() {
    let a = Var::new(3.0, "a");
    let b = &a - 1.;
    b.set_label("b");
    draw_dot(&b, "weights");

}

fn ex11() {
    let x1 = Var::new(2.0, "x1");
    let x2 = Var::new(0.0, "x2");
    let w1 = Var::new(-3.0, "w1");
    let w2 = Var::new(1.0, "w2");
    let b = Var::new(6.8812735870195432, "b");

    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;

    let x1w1x2w2 = &x1w1 + &x2w2;
    
    let n = &x1w1x2w2 + &b;
    n.set_label("n");


    let e = (2. * &n).exp();
    let o = &(&e - 1.) / &(&e + 1.); // Note: we have implemented ops for &Var
    

    o.set_label("o");

    o.back_propagate();

    draw_dot(&o, "weights_scrap");

}

fn ex12() {
    // Ignore this cell !!!!!!!
    // This is manual working of example given in cell - ex11
    let n = 0.8813_f64;
    let o1 = n.tanh();
    let do1dn = 1. - o1.powf(2.);
    println!("o1 = {}", o1);
    println!("do1/dn = {}", do1dn);
    // println!("o1 = {}", o1);
    // let nx2: f64 = 2. * n;
    // println!("2*n = {}", nx2);
    // let e2n = nx2.exp();
    // println!("e2n = {}", e2n);
    // let e2np1 = e2n + 1.;
    // println!("e2np1 = {}", e2np1);
    // let e2nm1 = e2n - 1.;
    // println!("e2nm1 = {}", e2nm1);
    // let e2npp1 = e2np1.powf(-1.); // raised to power of -1
    // println!("e2np1**-1 = {}", e2npp1);
    // let tanhn = e2nm1 * e2npp1;
    // println!("tanh = {}", tanhn);

    // println!("df/dd = {}", -1. * e2np1.powf(-2.));
    // println!("f = {}", e2np1.powf(-1.));

    // a = 2n
    // da/dn = 2
    let a = 2. * n;
    let da_dn = 2.;
    println!("a = {}", a);
    println!("da/dn = {}", da_dn);

    // b = e^a
    // db/da = e^a = 5.82757
    let b = a.exp();
    let db_da = b;
    println!("b = {}", b);
    println!("db/da = {}", b );

    // c = b - 1
    // dc/db = 1
    let c = b - 1.;
    let dc_db = 1.;
    println!("c = {}", c);
    println!("dc/db = {}", dc_db);

    // d = b + 1
    // dd/db = 1
    let d = b + 1.;
    let dd_db = 1.;
    println!("d = {}", d);
    println!("dd/db = {}", dd_db);

    // f = d ** -1
    // df/dd = -1 * (d ** -2)
    let f = d.powf(-1.);
    let df_dd = -1. * d.powf(-2.);
    println!("f = {}", f);
    println!("df/dd = {}", df_dd);

    // o = c * f
    let o = c * f;
    let do_dc = f;
    let do_df = c;
    println!("o = {}", o);
    println!("do/dc = {}", do_dc);
    println!("do/df = {}", do_df);

    let d_grad = 1.0 * do_df * df_dd;
    println!("dgrad = {}", d_grad);

    let c_grad = 1.0 * do_dc;
    println!("cgrad = {}", c_grad);

    // B gradient
    // b_grad_1 = do_do * do_df * df_dd * dd_db
    // let b_grad_1 = 1.0 * do_df * df_dd * dd_db;
    let b_grad_1 = d_grad * dd_db;

    // b_grad_2 = do_do * do_dc * dc_db
    // let b_grad_2 = 1.0 * do_dc * dc_db;
    let b_grad_2 = c_grad * dc_db;

    println!("b_grad_1 = {}", b_grad_1);
    println!("b_grad_2 = {}", b_grad_2);

    let b_grad = b_grad_1 + b_grad_2;
    println!("bgrad = {}", b_grad);

    let a_grad = b_grad * db_da;
    println!("a_grad = {}", a_grad);
    println!("debug: {}", 0.04290399662066896 * 5.827569394704068_f64 );

    // n_grad = db_db * db_da * da_dn
    let n_grad = b_grad * db_da * da_dn;
    println!("ngrad = {}", n_grad);
}

fn sanity_check() {
    let x = Var::new(-4.0, "");
    let a = 2.0_f64 * &x;
    let c = &a + 2.0;
    let z = &c + &x;
    let d = z.relu();
    let e = &d + &z;
    let q = &e * &x;
    let h = (&z * &z).relu();
    let f = &(&h + &q) + &q;
    let y = &f * &x;
    y.back_propagate();
    let (xmg, ymg) = (x, y);

    // Burn
    let x = tensor(&[-4.0]);
    let a = x.clone() * 2.0;
    let c = a + 2.0;
    let z = c + x.clone();
    let d = Relu::new().forward(z.clone());
    let e = d + z.clone();
    let q = e.clone() * x.clone();
    let h = Relu::new().forward(z.clone() * z.clone());
    let f = (h.clone() + q.clone()) + q.clone();
    let y = f.clone() * x.clone();
    let gradients = y.backward();
    let x_grad = grad(&x, &gradients);

    println!("Microgradr grad x = {}", xmg.grad());
    println!("Burn grad x = {}", x_grad);

    println!("Microgradr y = {}", ymg.data());
    println!("Burn y = {}", y.into_scalar());
}

fn test_more_ops() {
    let a = Var::new(-4.0, "");
    let b = Var::new(2.0, "");
    let c = &a + &b;
    let d = &(&a * &b) + &(b.pow(3.0));
    let c = &(&c + &c) + 1.0;
    let c1 = &(&c + 1.0) + &c;
    let c = &c1 + &(&a * -1.0);
    let d1 = &d + &(&d * 2.0);
    let d2 = (&b + &a).relu();
    let d = &d1 + &d2;
    let d1 = &d + &(&d * 3.0);
    let d2 = (&b - &a).relu();
    let d = &d1 + &d2; 
    let e = &c - &d;
    let f = e.pow(2.0);
    let g = &f * (1.0/2.0);
    let g1 = f.pow(-1.0);
    let g = &g + &(&g1 * 10.0);
    g.back_propagate();
    let (amg, bmg, gmg) = (a, b, g);

    // Burn
    let a = tensor(&[-4.0]);
    let b = tensor(&[2.0]);
    let c = a.clone() + b.clone();
    let d = (a.clone() * b.clone()) + b.clone().powf_scalar(3.0);
    let c = c.clone() + c.clone() + 1.0;
    let c = c.clone() + 1.0 + c.clone() + (a.clone().neg());
    let d1 = d.clone() + (d.clone() * 2.0);
    let d2 = Relu::new().forward(b.clone() + a.clone());
    let d = d1 + d2;
    let d1 = d.clone() + (d.clone() * 3.0);
    let d2 = Relu::new().forward(b.clone() - a.clone());
    let d = d1 + d2;
    let e = c.clone() - d.clone();
    let f = e.powf_scalar(2.0);
    let g = f.clone() / 2.0;
    let g = g.clone() + (f.powf_scalar(-1.0) * 10.0);
    let gradients = g.backward();

    println!("gmg = {}", gmg.data());
    println!("g = {}", g.into_scalar());

    let a_grad = grad(&a, &gradients);
    let b_grad = grad(&b, &gradients);

    println!("amg grad = {}", amg.grad());
    println!("a_grad = {}", a_grad);
    println!("bmg grad = {}", bmg.grad());
    println!("b_grad = {}", b_grad);

}

pub fn main() {
    println!("Notebook 3");
    test_more_ops();
}

// YT: 1:28
// VLC: 56.05