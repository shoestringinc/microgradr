#![allow(unused)]

use crate::numr::*;
use crate::plot::*;
use crate::engine::*;
use crate::graphviz::*;
use crate::nn::*;
use crate::utils::{rand_uniform_nums};
use std::iter::Sum;

fn ex1() {
    let a = Var::new(2.0, "a");
    // b = a^2
    // db/da = 2a = 4.0
    let b = a.pow(2.0);
    b.set_label("b");
    b.back_propagate();
    draw_dot(&b, "weights_burn");

    // Now instead of PyTorch and I run really an old Macbook intel, we use
    // Burn!
}

fn ex2() {
    let x = tensor(&[2.0]);
    let y = x.clone().powf_scalar(2.0);
    let gradients = y.backward();
    let x_grad = grad(&x, &gradients);
    println!("{}", x_grad);
}

fn ex3() {
    let x1 = tensor(&[2.0]);
    let x2 = tensor(&[0.0]);
    let w1 = tensor(&[-3.0]);
    let w2 = tensor(&[1.0]);
    let b = tensor(&[6.8812735870195432]);
    let n = x1.clone()*w1.clone() + x2*w2 + b;
    let o = n.tanh();

    println!("o is: {}", data(&o));

    let gradients = o.backward();
    let x1_grad = grad(&x1, &gradients);
    println!("x1_grad = {}", x1_grad);
    let w1_grad = grad(&w1, &gradients);
    println!("w1_grad = {}", w1_grad);
}

fn ex4() {
    let xs = rand_uniform_nums(3);
    println!("{:?}", xs);
}

fn ex5() {
    let a = Var::new(2., "");
    let b = Var::new(3., "");
    let n = Neuron::new(2);
    let o = n.output(&[a, b]);
}

fn ex6() {
    let n = Neuron::new(2);
    let inputs = [2., 3.];
    let inputs = inputs.map(|v|Var::new(v, ""));
    let o = n.output(&inputs);
    dbg!(&o);
    println!("o is: {}", o.data());
    draw_dot(&o, "big_net");

}

fn ex7() {
    let inputs = [2., 3.];
    let n = Layer::new(2, 3);
    let res = n.output_prims(&inputs);
    res.iter().for_each(|v|println!("{}", v)); 
    dbg!(&res[0]);
}

fn ex8() {
    let inputs = [2., 3., -1.];
    let mlp = MLP::new(3, &[4, 4, 1]);
    let res = mlp.output_prims(&inputs);
    res.iter().for_each(|e|println!("{}", e));
    let o = &res[0];
    // dbg!(o);
    draw_dot(o, "big_net");
}

fn ex9() {

    let mlp = MLP::new(3, &[4, 4, 1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    // desired targets for each of the inputs from the above set
    let ys = vec![1.0, -1.0, -1.0, 1.0];

    let ypred: Vec<_> = xs.iter()
        .map(|e|mlp.output_prims(e))
        .map(|e|e[0].clone()) // get the final layer neuron
        .collect();

    // ypred.iter().for_each(|e|println!("{}", e));

    // loss calculation
    let losses: Vec<_> = ys.iter().zip(&ypred)
        .map(|(&x1, x2)| (x2 - x1).pow(2.0))
        .collect();
    println!("losses");
    losses.iter().for_each(|e|println!("{}", e));
    // let first_diff = &ypred[0] + ys[0]*-1.0;
    // println!("first loss: {}", first_diff.pow(2.0));
    let loss = losses.iter().cloned().reduce(|e1, e2|&e1 + &e2).unwrap();

    // layer[0].neuron[0].weight[0]
    let w = mlp.layer(0).neuron(0).weight(0);
    // draw_dot(&loss, "big_net");
    println!("loss is: {}", loss);
    println!("w = {}", w);
    println!("w.grad = {}", w.grad());
    loss.back_propagate();
    println!("w.grad = {}", w.grad());

}

fn calc_loss(mlp: &MLP, xs: &Vec<Vec<f64>>, ys: &Vec<f64>) -> Var {
    let ypred: Vec<_> = xs.iter()
        .map(|e|mlp.output_prims(e))
        .map(|e|e[0].clone()) // get the final layer neuron
        .collect();

    // loss calculation
    let losses: Vec<_> = ys.iter().zip(&ypred)
        .map(|(&x1, x2)| (x2 - x1).pow(2.0))
        .collect();

    let loss = losses.iter().cloned().reduce(|e1, e2|&e1 + &e2).unwrap();

    loss
}

fn zero_grad(params: &Vec<Var>) {
    for p in params {
        p.set_grad(0.0);
    }
}

fn ex10() {
    let mlp = MLP::new(3, &[4, 4, 1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    // desired targets for each of the inputs from the above set
    let ys = vec![1.0, -1.0, -1.0, 1.0];

    let loss = calc_loss(&mlp, &xs, &ys);

    let params = mlp.parameters();
    params.iter().for_each(|e|println!("{}", e));
    println!("Total number of parameters: {}", params.len());

    // layer[0].neuron[0].w[0]
    let w = mlp.layer(0).neuron(0).weight(0);
    println!("w = {}", w.data());
    println!("w = {}", &params[0].data());
    println!("w grad before back propagation = {}", w.grad());
    println!("loss before back propagation = {}", loss.data());
    loss.back_propagate();
    println!("w grad after back propagation = {}", w.grad());
    let nudge = 0.005;
    if w.grad() > 0. {
        w.set_data(w.data() - nudge);
    } else {
        w.set_data(w.data() + nudge);
    }
    // loss.reset_grads(); // this is inefficient as we should zero only params!
    zero_grad(&params);
    println!("w grad after reset = {}", w.grad());
    let loss = calc_loss(&mlp, &xs, &ys);
    println!("loss after fine tuning w = {}", loss.data());
    

}

fn ex11() {
    let mlp = MLP::new(3, &[4, 4, 1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    // desired targets for each of the inputs from the above set
    let ys = vec![1.0, -1.0, -1.0, 1.0];

    let loss = calc_loss(&mlp, &xs, &ys);

    let params = mlp.parameters();

    let w = mlp.layer(0).neuron(0).weight(0);

    loss.back_propagate();

    println!("w grad = {}", w.grad());

    w.set_data(w.data() + (-0.01 * w.grad()));

    println!("loss before tuning = {}", loss);

    let loss = calc_loss(&mlp, &xs, &ys);

    println!("loss after tuning = {}", loss);



}

fn ex12() {
    let mlp = MLP::new(3, &[4, 4, 1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    // desired targets for each of the inputs from the above set
    let ys = vec![1.0, -1.0, -1.0, 1.0];

    let params = mlp.parameters();

    // Iterations for seeing the loss minimization using gradient descent
    // This is gradient descent in action!
    (1..=20).for_each(|i|{

        // forward pass
        let loss = calc_loss(&mlp, &xs, &ys);
        
        // backward pass
        zero_grad(&params);
        loss.back_propagate();

        // update
        for p in &params {
            p.set_data(p.data() + (-0.05 * p.grad()));
        }

        println!("Loss #{} = {}", i,  loss.data() );
    });
}

pub fn main() {
    println!("Notebook 4");
    ex12();
}

// VLC: 37:48