#![allow(unused)]

use std::collections::VecDeque;

use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::backend::autodiff::grads::Gradients;
use burn::prelude::*;

use crate::engine::*;
use crate::graphviz::draw_dot;
use crate::utils::*;

// Backend 
type Bck = Autodiff<NdArray<f64>>;

const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

type TensorType = Tensor<Autodiff<NdArray<f64>>, 1>;

pub fn tensor(data: &[f64]) -> TensorType {
    let t = Tensor::<Bck, 1>::from_data(data, &DEVICE).require_grad();
    t
}

pub fn data(t: &TensorType) -> f64 {
    t.clone().into_scalar()
}

pub fn grad(wrt: &TensorType, gradients: &Gradients) -> f64 {
    let g = wrt.grad(gradients).unwrap();
    g.into_scalar()
}

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<Var>,
    bias: Var,
}

impl Neuron {

    pub fn new(num_inputs: usize) -> Self {
        let weights = rand_uniform_nums(num_inputs)
            .into_iter()
            .map(|v| Var::new(v, ""))
            .collect();

        let bias = Var::new(rand_uniform(), "");
        Self {
            weights,
            bias,
        }
    }

    pub fn output(&self, inputs: &[Var]) -> Var {
        // weights * inputs + bias
        assert_eq!(inputs.len(), self.weights.len());
        let res: Vec<_> = self.weights.iter().zip(inputs)
            .map(|(w, x)| w * x).collect();
        let sum_wx = res.iter().cloned()
            .reduce(|e1, e2|&e1 + &e2).unwrap();
        let n = &sum_wx + &self.bias;
        n.tanh()        
    }

    pub fn output_prims(&self, inputs: &[f64]) -> Var {
        let inputs: Vec<_> = inputs.iter()
            .map(|&e| Var::new(e, "")).collect();
        self.output(&inputs)
    }

    pub fn weight(&self, n: usize) -> Var {
        self.weights[n].clone()
    }

    pub fn parameters(&self) -> Vec<Var> {
        let mut ws = self.weights.clone();
        ws.push(self.bias.clone());
        ws
    }
    
}

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_neurons: usize) -> Self {
        let neurons = (1..=num_neurons)
            .map(|_|Neuron::new(num_inputs)).collect();
        Self {
            neurons
        }
    }

    pub fn output(&self, inputs: &[Var]) -> Vec<Var> {
        let outs = self.neurons.iter()
            .map(|n|n.output(inputs)).collect();
        outs
    }

    pub fn output_prims(&self, inputs: &[f64]) -> Vec<Var> {
        let inputs: Vec<_> = inputs.iter()
            .map(|&e|Var::new(e,  "")).collect();
        self.output(&inputs)
    }

    pub fn neuron(&self, n: usize) -> Neuron {
        self.neurons[n].clone()
    }

    pub fn parameters(&self) -> Vec<Var> {
        let mut result = Vec::new();
        for n in &self.neurons {
            result.extend_from_slice(&n.parameters());
        }   
        result
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(num_inputs: usize, layers_with_neurons: &[usize]) -> Self {
        let mut sz = VecDeque::from_iter(layers_with_neurons);
        sz.push_front(&num_inputs);
        let layers = 
            (0..layers_with_neurons.len())
                .map(|i| Layer::new(*sz[i], *sz[i+1]))
                .collect();
        Self {
            layers
        }
    }

    pub fn output(&self, inputs: &[Var]) -> Vec<Var> {
        let res: Vec<_> = self.layers.iter()
            .fold(inputs.to_vec(), |inp, e| {
                e.output(&inp)
            });
        res
    }

    pub fn output_prims(&self, inputs: &[f64]) -> Vec<Var> {
        let inputs: Vec<_> = inputs.iter()
            .map(|&e|Var::new(e, "")).collect();
        self.output(&inputs)
    }

    pub fn layer(&self, n: usize) -> Layer {
        self.layers[n].clone()
    }

    pub fn parameters(&self) -> Vec<Var> {
        let mut result = Vec::new();
        for l in &self.layers {
            result.extend_from_slice(&l.parameters());
        }
        result
    }
}

/*
Example code:

    let device = NdArrayDevice::Cpu;
    let x = Tensor::<Bck, 1>::from_data([2.0], &device).require_grad();
    let y = x.clone().powf_scalar(2.0);
    let loss = y.sum();
    let mut gradients = loss.backward();
    let x_grad = x.grad(&gradients).unwrap();
    dbg!(&x_grad);
    let v = x_grad.to_data();
    let res = v.to_vec::<f64>().unwrap();
    println!("{:?}", res);

Better stuff:
    let device = NdArrayDevice::Cpu;
    let x = Tensor::<Bck, 1>::from_data([2.0], &device).require_grad();
    let y = x.clone().powf_scalar(2.0);
    let mut gradients = y.backward();
    let x_grad = x.grad(&gradients).unwrap();
    println!("{}", x_grad.into_scalar());


 */