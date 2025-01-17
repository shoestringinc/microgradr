# microgradr

### Objective

This is a rewrite of Andrej Karpathy's Autograd engine in Rust programming language.

Original repo is at:  
<https://github.com/karpathy/micrograd>

The superb video that taught maths, autograd and neural net is at:  
<https://www.youtube.com/watch?v=VMj-3S1tku0>

The objective was to demonstrate internally in our company [Shoestring Software](https://www.shoestringinc.com/) that we can do an almost one to one Pythonic development both at engine level and at the user level(i.e. notebook) using Rust. 

The challenge was to demonstrate the following: 
- **normal flow of programming** without getting bogged down by famous borrow checker.
- can be used by Python developers (at notebook level) with under 1 day intro to Rust!
- code has to be written in functional style (so that Clojure and Scala guys like it).
- code looks pretty in the first uncut version itself!
- If possible, lifetime annotations should not be sticking all over!
- The engine can be hacked upon by our juniour folks.

If the challenge could be met than everyone decided we will switch over to Rust for everything and not just our tooling and infrastructure level software.

By everything we mean:
- Scripting
- Frontend app development including mobile.
- Backend app development
- anything else imaginable.

We met all the objectives and [Shoestring Software](https://www.shoestringinc.com/) became an all [Rust](https://www.rust-lang.org/) company.

### Prerequisites

You must have Rust installed. Follow instructions here:  
<https://www.rust-lang.org/tools/install>

Install [Graphviz](https://graphviz.org/). Installation information is at:  
<https://graphviz.org/download/>

### Installation

Clone this repo in any folder.

Inside the folder, just do:

```bash
cargo run
```

### Code layout

The code has been written as a single binary crate with modules for libs as well as *notebook* explorations.

It can be refactored into a library crate with multiple binaries for each of the notebook later.

The *notebook* files are prefixed with `nb`. For e.g. `nb1`, `nb2`, etc.

Each of the *cell* inside a *notebook* file is written as a Rust function prefixed with `ex`.  
For e.g. `ex1`, `ex2`, etc.

### Usage

Whenever you want to execute cells in a particular notebook, do the following:  
- go to `main.rs`.
- In the `main` function, select the notebook (for e.g. `nb3`) by:    
```rust
nb3::main();
```
- Each notebook file has its own `main` function.
- Inside the `nb3.rs` file, execute any cell (for e.g. `ex3`) by executing the `ex3()` function in `nb3` `main` function:

### Code examples

The main modules are `engine` and `nn`.

The main structure is called `Var` and not `Value` as in Andrej's code. In our Rust code, it wraps over `Value` struct.

Checkout `nb3` for `engine`(i.e. autograd) use.  
Checkout `nb4` for `nn`(i.e. neural net) use.

Remember in Rust we have implemented all the numeric operations traits (addition, multiplication etc.) on `&Var` and not `Var`.

This keeps the notebook level code clean as we don't have to clone incessantly (though cloning of `Var` is inexpensive). User types much less.

A simple example:

```rust
let a = Var::new(2.0, "a");
let b = Var::new(3.0, "b");
let c = &a + &b;
c.set_label("c");
c.back_propate();
// print value
println!("{}", c.data());
// print gradient
println!("{}", a.grad());
```

### Important imports for notebooks

```rust
use crate::engine::*;
use crate::numr::*;
use crate::plot::*;
use crate::utils::*;
use crate::graphviz::*;
use crate::nn::*;
```

### Plotting

Andrej uses matplotlib for plotting.

We have a simple function `plot_to_file()` in `plot` module.

It is a simple wrapper over `plotters` crate. 

An example usage:

```rust
let xs = arange(-5., 5., 0.25);
let ys = xs.mapv(f);
plot_to_file(&xs, &ys, "plot.png");
```

Open the files in any rendering app. We use `Preview` on OSX. `Preview` automatically re-renders as the graph changes.

### Graph Visualization

We generate dot files directly (no crates are used). Ensure `Graphviz` is installed.

To get a graph visualization:

```rust
    draw_dot(&c, "weights");
```
Note: we should not provide extension here as 2 files are generated: `.dot` and `.png`.

### Testing

Andrej provides two tests in the `test_engine.py` file. 

He does sanity check against `PyTorch`. We do the same against Rust [Burn lib](https://burn.dev/).

However, the two functions are not under `tests` folder but are the end of `nb3` file.

BTW, it was amazing in the very first cut to see Burn and Microgradr give identical result!

### Final notes

Lots of reorgnization and refactoring can be done including separate test folder, lib and binaries etc.

### License

MIT 

See the LICENSE file for more info.
