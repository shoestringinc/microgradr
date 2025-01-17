#![allow(unused)]

// This generates dot files that allows you to visual graphs using graphviz program.
// Please install graphviz.
// There is no supporting crate. 
// I just wrote what Andrej did in Python.

use crate::engine::*;
use std::collections::{HashMap, HashSet};
use std::process::Command;
use std::fs;

fn trace(root: &Var) -> (HashSet<(u128, u128)>, HashMap<u128, Var>) {
    let mut edges = HashSet::new();
    let mut nodes = HashMap::new();

    fn build(var: &Var, edges: &mut HashSet<(u128, u128)>, nodes: &mut HashMap<u128, Var>) {
        let uid = var.uid();
        let children = &var.children();
        if !nodes.contains_key(&uid) {
            nodes.insert(uid, var.clone());
            for child in children {
                edges.insert((child.uid(), uid));
                build(child, edges, nodes);
            }
        }
    }

    build(root, &mut edges, &mut nodes);

    (edges, nodes)
}

// NOTE: file_name is minus extension
// as we use it for both dot file and png file.
pub fn draw_dot(var: &Var, file_name: &str)  {
    let (edges, nodes) = trace(var);

    let mut body = String::new();

    // Render the nodes (note we are producing dot files to be rendered by graphviz)
    for (k, node) in &nodes {
        // If you want shorter labels
        // let label: String = node.label().chars().take(5).collect();
        let label = node.label().clone();
        let item = format!(r#"{} [label = "|{}| data {:.4}, grad: {}, op: {}"] "#, 
        node.uid(), label, node.data(), node.grad(), node.op() );
        body.push_str(&item);
        body.push_str("\n");
    }

    // Render edges 
    for (k, v) in &edges {
        let item = format!("{} -> {}", k, v);
        body.push_str(&item);
        body.push_str("\n");
    }

    let content = format!("digraph {{\n {} \n}}", body);
    // println!("{content}");

    let dot_file_name = format!("{}.dot", file_name);
    let graph_file_name = format!("{}.png", file_name);
    fs::write(&dot_file_name, &content);
    Command::new("dot")
        .args(["-Tpng", &dot_file_name, "-o", &graph_file_name])
        .status().unwrap();


}