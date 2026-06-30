// Chapter 11. A full transformer block (pre-LN, with residual connections) — in pure Swift.
//
// We assemble everything from Part III into one block:
//   h   = x + Attention(LayerNorm(x))      // sublayer 1: "communication" between words
//   out = h + FFN(LayerNorm(h))            // sublayer 2: "reflection" inside a word
// The residual connections (x + ...) let the gradient flow through dozens of such blocks.
//
// Run:
//     swift ch11_block.swift
//
// The production version is in the companion repository: mini-gpt/Sources/MiniGPT/GPT.swift

import Foundation

let d = 4
let eps = 1e-5

// --- vector addition (needed for the residual connections) ---
func vadd(_ a: [Double], _ b: [Double]) -> [Double] { zip(a, b).map { $0.0 + $0.1 } }

// --- LayerNorm: bring the vector to zero mean and unit variance ---
func layerNorm(_ v: [Double]) -> [Double] {
    let mean = v.reduce(0.0, +) / Double(v.count)
    let varc = v.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(v.count)
    let denom = (varc + eps).squareRoot()
    return v.map { ($0 - mean) / denom }   // gamma = 1, beta = 0 for simplicity
}

// --- causal self-attention (from Chapter 8), projections = identity ---
func dot(_ a: [Double], _ b: [Double]) -> Double { zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 } }
func softmax(_ xs: [Double]) -> [Double] {
    let m = xs.max()!; let e = xs.map { exp($0 - m) }; let s = e.reduce(0.0, +)
    return e.map { $0 / s }
}
func causalSelfAttention(_ X: [[Double]]) -> [[Double]] {
    let scale = 1.0 / Double(d).squareRoot()
    return (0..<X.count).map { i in
        let scores = (0...i).map { j in dot(X[i], X[j]) * scale }
        let w = softmax(scores)
        var o = [Double](repeating: 0.0, count: d)
        for j in 0...i { for k in 0..<d { o[k] += w[j] * X[j][k] } }
        return o
    }
}

// --- feed-forward (from Chapter 9) ---
func gelu(_ x: Double) -> Double { 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))) }
func linear(_ x: [Double], _ W: [[Double]], _ b: [Double]) -> [Double] {
    zip(W, b).map { row, bias in zip(x, row).reduce(bias) { $0 + $1.0 * $1.1 } }
}
let dFF = 16
func wv(_ i: Int, _ j: Int) -> Double { sin(Double(i * 7 + j * 3)) * 0.3 }
let W1 = (0..<dFF).map { i in (0..<d).map { j in wv(i, j) } }
let W2 = (0..<d).map { i in (0..<dFF).map { j in wv(i, j) } }
let b1 = [Double](repeating: 0, count: dFF), b2 = [Double](repeating: 0, count: d)
func ffn(_ x: [Double]) -> [Double] { linear(linear(x, W1, b1).map(gelu), W2, b2) }

// --- ONE TRANSFORMER BLOCK ---
func transformerBlock(_ X: [[Double]]) -> [[Double]] {
    let attn = causalSelfAttention(X.map(layerNorm))      // Attention(LN(x))
    let h = zip(X, attn).map { vadd($0.0, $0.1) }         // x + ...  (residual connection)
    let ff = h.map(layerNorm).map(ffn)                    // FFN(LN(h))
    return zip(h, ff).map { vadd($0.0, $0.1) }            // h + ...  (residual connection)
}

let X: [[Double]] = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]
print("Input (3 tokens, d=4):")
for (i, v) in X.enumerated() { print("  token \(i + 1): \(v)") }
print("Block output (same 3×4 shape, but with context taken into account):")
for (i, v) in transformerBlock(X).enumerated() {
    print("  token \(i + 1): \(v.map { ($0 * 1000).rounded() / 1000 })")
}
