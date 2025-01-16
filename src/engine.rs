#![allow(unused)]

use uuid::Uuid;
use std::collections::HashSet;
use std::{u128};

use std::cell::RefCell;
use std::rc::Rc;

use std::ops::{Add, Div, Mul, Neg, Sub};

use std::iter::Sum;

fn uuid_to_u128() -> u128 {
    let uuid = Uuid::new_v4();
    let n = u128::from_be_bytes(*uuid.as_bytes());
    n
}

#[derive(Debug)]
struct Value {
    // also uid is used for giving node ids in graphviz
    // had to use this because we can't have Hash, Eq out of f64, if used for Set operations
    pub uid: u128, 
    pub data: f64,
    pub grad: f64,          // impact on the say loss function by default is 0.0
    pub children: Vec<Var>, // expression graph. 
    pub op: String,
    pub label: String,
}

impl Value {
    pub fn new(data: f64, label: &str) -> Self {
        Value {
            uid: uuid_to_u128(),
            data,
            grad: 0.0,
            children: vec![],
            op: " ".to_string(),
            label: label.into(),
        }
    }
}

type NodeRef = Rc<RefCell<Value>>;

#[derive(Debug, Clone)]
pub struct Var {
    value: NodeRef,
}
// cheap clone - only a pointer Rc

impl Var {
    pub fn new(data: f64, label: &str) -> Self {
        let v = Value::new(data, label.into());
        Var {
            value: Rc::new(RefCell::new(v)),
        }
    }

    pub fn uid(&self) -> u128 {
        self.value.borrow().uid
    }

    pub fn label(&self) -> String {
        self.value.borrow().label.clone()
    }

    pub fn set_label(&self, label: &str) {
        self.value.borrow_mut().label = label.into();
    }

    pub fn grad(&self) -> f64 {
        self.value.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.value.borrow_mut().grad = grad;
    }

    pub fn data(&self) -> f64 {
        self.value.borrow().data
    }

    pub fn set_data(&self, new_data: f64) {
        self.value.borrow_mut().data = new_data;
    }

    pub fn op(&self) -> String {
        self.value.borrow().op.clone()
    }

    pub fn pow(&self, n: f64) -> Var {
        let data = self.data().powf(n);
        let npower = Var::new(n, &n.to_string());
        let mut v = Value::new(data, "");
        v.op = "pow".into();
        v.label = format!("{} ** {:.4}", self.label(), n);
        v.children = vec![self.clone(), npower]; // we will require npower in grad calculation
        Var {
            value: Rc::new(RefCell::new(v))
        }
    }

    pub fn exp(&self) -> Self {
        let data = self.data().exp();
        let mut v = Value::new(data, "");
        v.op = "exp".into();
        v.label = format!("exp({})", self.label());
        v.children = vec![self.clone()];
        Var {
            value: Rc::new(RefCell::new(v))
        }
    }

    pub fn relu(&self) -> Self {
        let new_data = if self.data() < 0.0 { 0.0 } else { self.data() };
        let mut v = Value::new(new_data, "");
        v.op = "relu".into();
        v.label = format!("relu({})", self.label());
        v.children = vec![self.clone()];
        Var {
            value: Rc::new(RefCell::new(v))
        }
    }

    pub fn tanh(&self) -> Self {
        // let x = self.value.borrow().data;
        let x = self.data();

        // We can simply calculate as follows:
        // let data = x.tanh();

        // But, Andrej uses the following formula
        // tanh(x) = e^2x -1 / e^2x + 1
        let x1 = 2. * x;
        let e2x = x1.exp();
        let data = (e2x - 1.) / (e2x + 1.);

        let mut v = Value::new(data, "");
        v.op = "tanh".into();
        v.label = format!("tanh({})", self.label());
        v.children = vec![self.clone()];
        Var {
            value: Rc::new(RefCell::new(v))
        }
    }

    // cheap cloning as Var only has a Rc.
    pub fn children(&self) -> Vec<Var> {
        self.value.borrow().children.clone()
    }

    pub fn dump_child_labels(&self) {
        let children = &self.value.borrow().children;
        println!("Children of Node: {}", self.label());
        for child in children {
            println!("{}", child.label());
        }
        println!("------");
    }

    fn topological_sort(&self) -> Vec<Var> {
        let mut topo = vec![];
        let mut visited = HashSet::new();

        fn build_topo(v: &Var, visited: &mut HashSet<u128>, topo: &mut Vec<Var>) {
           if !visited.contains(&v.uid()) {
                visited.insert(v.uid());
                for child in v.children() {
                    build_topo(&child, visited, topo);
                }
                topo.push(v.clone());
           }
        }

        build_topo(self, &mut visited, &mut topo);
        topo
    }

    // pub fn reset_grads(&self) {
    //     let topo = self.topological_sort();
    //     for node in topo.iter() {
    //         node.set_grad(0.);
    //     }
    // }

    pub fn back_propagate(&self) {
        // let mut topo = vec![];
        // let mut visited = HashSet::new();

        // fn build_topo(v: &Var, visited: &mut HashSet<u128>, topo: &mut Vec<Var>) {
        //    if !visited.contains(&v.uid()) {
        //         visited.insert(v.uid());
        //         for child in v.children() {
        //             build_topo(&child, visited, topo);
        //         }
        //         topo.push(v.clone());
        //    }
        // }

        // build_topo(self, &mut visited, &mut topo);

        let topo = self.topological_sort();

        self.set_grad(1.);
        for node in topo.iter().rev() {
            node.backward();
        }
    }

    fn backward(&self) {
        let outer_grad = self.grad();
        let children = &self.value.borrow().children;
        match self.op().as_str() {
            "+" => {
                let first = &children[0];
                let second = &children[1];
                let new_grad = outer_grad * 1.0;
                first.set_grad(first.grad() + new_grad);
                second.set_grad(second.grad() + new_grad);
            }
            "*" => {
                let first = &children[0];
                let second = &children[1];
                let new_first_grad = second.data() * outer_grad;
                first.set_grad(first.grad() + new_first_grad);
                let new_second_grad = first.data() * outer_grad;
                second.set_grad(second.grad() + new_second_grad);
            }
            "tanh" => {
                let child = &children[0];
                // d(tanh(x))/dx = 1 - (tanh(x))^2
                let x = self.data().powf(2.0); // Note: self.data = tanh(child.data)
                let local_grad = 1. - x;
                let global_grad = local_grad * outer_grad;

                child.set_grad(child.grad() + global_grad);
            }
            "relu" => {
                let child = &children[0];
                let local_grad = if child.data() > 0.0 { 1.0 } else { 0.0 };
                let global_grad = local_grad * outer_grad;

                child.set_grad(child.grad() + global_grad);

            }
            "exp" => {
                let child = &children[0];
                // d(e^x)/dx = e^x
                let local_grad = self.data();
                let global_grad = local_grad * outer_grad;
                child.set_grad(child.grad() + global_grad);
            }
            "pow" => {
                let child = &children[0];
                let n = &children[1];
                let power = n.data();
                // println!("power: {}", power);
                // f = x**n
                // df/dx = n * (x ** (n-1))
                let local_grad = power * (child.data().powf(power - 1.));
                let global_grad = local_grad * outer_grad;
                child.set_grad(child.grad() + global_grad);


            }
            _ => {}
        }

    }
}

impl Add for &Var {
    type Output = Var;

    fn add(self, rhs: Self) -> Self::Output {
        let first = self.value.borrow();
        let second = rhs.value.borrow();
        let data = first.data + second.data;
        let mut v = Value::new(data, " ");
        v.children = vec![self.clone(), rhs.clone()];
        v.op = "+".into();
        v.label = format!("({}{}{})", first.label, "+", second.label);
        Var {
            value: Rc::new(RefCell::new(v)),
        }
    }
}

impl Add<f64> for &Var {
    type Output = Var;

    fn add(self, rhs: f64) -> Self::Output {
        let second = Var::new(rhs, &rhs.to_string());
        self + &second
    }
    
}

impl Add<&Var> for f64 {
    type Output = Var;

    fn add(self, rhs: &Var) -> Self::Output {
        let first = Var::new(self, &self.to_string());
        &first + rhs
    }
}


impl Mul for &Var {
    type Output = Var;

    fn mul(self, rhs: Self) -> Self::Output {
        let first = self.value.borrow();
        let second = rhs.value.borrow();
        let data = first.data * second.data;
        let mut v = Value::new(data, " ");
        v.children = vec![self.clone(), rhs.clone()];
        v.op = "*".into();
        v.label = format!("({}{}{})", first.label, "*", second.label);
        Var {
            value: Rc::new(RefCell::new(v)),
        }
        
    }
}

impl Mul<f64> for &Var {
    type Output = Var;

    fn mul(self, rhs: f64) -> Self::Output {
        let second = Var::new(rhs, &rhs.to_string());
        self * &second
    }
    
}

impl Mul<&Var> for f64 {
    type Output = Var;

    fn mul(self, rhs: &Var) -> Self::Output {
        let first = Var::new(self, &self.to_string());
        &first * rhs
    }
}

impl Div for &Var {
    type Output = Var;

    fn div(self, rhs: Self) -> Self::Output {
        let first = self.value.borrow();
        let rhs1 = rhs.pow(-1.);
        let mut v = self * &rhs1;
        v
        // let second = rhs.value.borrow();
        // let data = first.data * second.data;
        // let mut v = Value::new(data, " ");
        // v.children = vec![self.clone(), rhs.clone()];
        // v.op = "*".into();
        // v.label = format!("({}{}{})", first.label, "*", second.label);
        // Var {
        //     value: Rc::new(RefCell::new(v)),
        // }
        
    }
    
}

impl Neg for &Var {
    type Output = Var;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}

impl Sub for &Var {
    type Output = Var;

    fn sub(self, rhs: Self) -> Self::Output {
        let rhs1 = -rhs;
        self + &rhs1
    }
}

impl Sub<f64> for &Var {
    type Output = Var;

    fn sub(self, rhs: f64) -> Self::Output {
        let rhs1 = Var::new(rhs, &rhs.to_string());
        let rhs2 = -&rhs1;
        self + &rhs2
    }
    
}

impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Var(data={})", self.data())
    }
}
