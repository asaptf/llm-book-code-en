// Chapter 9. The transformer's feed-forward network (FFN) — in pure Swift.
//
// The FFN processes EACH token separately: it expands its vector (usually ×4),
// passes it through the GELU nonlinearity, and shrinks it back. This, it's believed,
// is where the lion's share of the model's "knowledge" is stored.
//
// Run:
//     swift ch09_ffn.swift
//
// The production version is in the companion repository: mini-gpt/Sources/MiniGPT/GPT.swift

import Foundation

// GELU — a smooth activation (tanh approximation), as in GPT.
// Similar to ReLU, but with no kink: negative values are dampened smoothly.
func gelu(_ x: Double) -> Double {
    0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
}

// Linear layer: y_i = (row W_i · x) + b_i. Output size = the number of rows of W.
func linear(_ x: [Double], _ W: [[Double]], _ b: [Double]) -> [Double] {
    zip(W, b).map { row, bias in zip(x, row).reduce(bias) { $0 + $1.0 * $1.1 } }
}

let dModel = 4      // the token vector's dimension
let dFF = 16        // the "expanded" inner dimension (×4)

// We fill the weights with a deterministic formula (instead of random) — for reproducibility.
func w(_ i: Int, _ j: Int) -> Double { sin(Double(i * 7 + j * 3)) * 0.3 }
let W1 = (0..<dFF).map    { i in (0..<dModel).map { j in w(i, j) } }   // dFF × dModel
let b1 = [Double](repeating: 0.0, count: dFF)
let W2 = (0..<dModel).map { i in (0..<dFF).map    { j in w(i, j) } }   // dModel × dFF
let b2 = [Double](repeating: 0.0, count: dModel)

// FFN(x) = W2 · GELU(W1·x + b1) + b2
func ffn(_ x: [Double]) -> [Double] {
    let hidden = linear(x, W1, b1).map(gelu)   // expand to dFF and apply GELU
    return linear(hidden, W2, b2)              // shrink back to dModel
}

// Show the shape of the GELU activation.
print("GELU softly suppresses negatives:")
for x in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
    print("  gelu(\(String(format: "%+.1f", x))) = \(String(format: "%+.3f", gelu(x)))")
}

let x = [1.0, 0.5, -0.5, 2.0]
print("\nFFN input:  \(x)")
print("FFN output: \(ffn(x).map { ($0 * 1000).rounded() / 1000 })")
