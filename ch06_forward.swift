// Chapter 6. The forward pass of a tiny neural network — in pure Swift.
//
// A two-layer network: 3 inputs -> a hidden layer of 4 neurons (ReLU) -> 1 output.
// No ML libraries: every multiplication is visible.
//
// Run:
//     swift ch06_forward.swift

import Foundation

/// ReLU activation: zero out the negative, pass the positive through.
func relu(_ x: Double) -> Double { max(0.0, x) }

/// Identity activation — "leave as is" (for the output layer).
func identity(_ x: Double) -> Double { x }

/// One neuron: weighted sum of inputs + bias (no activation).
func neuron(_ x: [Double], _ w: [Double], _ b: Double) -> Double {
    zip(x, w).reduce(0.0) { $0 + $1.0 * $1.1 } + b
}

/// Layer: apply each neuron (a row of W) to the input x, then the activation.
func layer(_ x: [Double], _ W: [[Double]], _ b: [Double],
           _ activation: (Double) -> Double) -> [Double] {
    zip(W, b).map { activation(neuron(x, $0.0, $0.1)) }
}

// --- The network's weights (in reality they're learned; here set by hand) ---
let W1: [[Double]] = [
    [0.5, -0.2, 0.1],
    [0.3, 0.8, -0.5],
    [-0.4, 0.2, 0.9],
    [0.1, 0.1, 0.1],
]
let b1 = [0.0, 0.1, -0.2, 0.05]

let W2: [[Double]] = [[0.7, -0.6, 0.2, 1.0]]   // one output neuron, 4 weights
let b2 = [0.0]

/// Forward pass: input -> hidden layer (ReLU) -> output (no activation).
func forward(_ x: [Double]) -> (h: [Double], y: [Double]) {
    let h = layer(x, W1, b1, relu)
    let y = layer(h, W2, b2, identity)
    return (h, y)
}

let x = [1.0, 2.0, -1.0]
let (h, y) = forward(x)
func r(_ v: [Double]) -> [Double] { v.map { ($0 * 1000).rounded() / 1000 } }
print("Input:          \(x)")
print("Hidden layer:   \(r(h))")
print("Network output: \(r(y))")
