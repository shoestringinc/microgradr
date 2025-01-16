#![allow(unused)]

// This is for plotting graphs like matplotlib.
// We use plotters crate for actual rendering.

use plotters::prelude::*;
use crate::numr::*;
use ndarray::prelude::*;
use ndarray::Array;

pub fn plot_to_file(xs: &Array1<f64>, ys: &Array1<f64>, file_name: &str) {
    let xmin = min_array1(xs);
    let xmax = max_array1(xs) + 0.1;
    let xrange = xmin..xmax;

    let ymin = min_array1(ys);
    let ymax = max_array1(ys) + 0.1;
    let yrange = ymin..ymax;

    let root = BitMapBackend::new(file_name, (640, 480));
    let draw = root.into_drawing_area();
    draw.fill(&WHITE);

    let mut chart = ChartBuilder::on(&draw)
        .caption("Simple Plot", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(20)
        .y_label_area_size(20)
        // .build_cartesian_2d(xs[0]..(xs[xs.len()-1] + 0.1), (ys[ys.len()-1])..(ys[0]+0.1))
        .build_cartesian_2d(xrange, yrange)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))
        .unwrap();
}


