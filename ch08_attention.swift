// Chapter 8. Self-attention with a causal mask — in pure Swift.
//
// We show the heart of the transformer on three toy tokens: how each word
// gathers context by comparing its query (Q) with the keys (K) of the other words
// and summing their values (V) with attention weights. The causal mask forbids
// "peeking into the future" — exactly as during generation in GPT.
//
// Run:
//     swift ch08_attention.swift
//
// The full multi-head attention implementation is in the companion repository:
//   mini-gpt/Sources/MiniGPT/GPT.swift

import Foundation

// 3 tokens, embeddings of dimension d = 4 (toy).
let X: [[Double]] = [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
]
let n = X.count
let d = X[0].count
let scale = 1.0 / Double(d).squareRoot()   // the divisor 1/√d for stability

// The Q, K, V projections. To focus on the idea of attention itself, we take
// identity matrices → Q = K = V = X. In a real model these are learnable
// matrices Wq, Wk, Wv (see the engineering aside in the chapter).
let Q = X, K = X, V = X

func dot(_ a: [Double], _ b: [Double]) -> Double {
    zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
}

func softmax(_ xs: [Double]) -> [Double] {
    let m = xs.max()!
    let e = xs.map { exp($0 - m) }
    let s = e.reduce(0.0, +)
    return e.map { $0 / s }
}

// For each token i we compute attention over tokens j (causal mask: j <= i).
for i in 0..<n {
    // 1. score(i, j) = (Q_i · K_j) / √d — but only for j <= i (don't look ahead).
    var scores: [Double] = []
    for j in 0...i { scores.append(dot(Q[i], K[j]) * scale) }

    // 2. softmax over the available positions → attention weights (sum to 1).
    let weights = softmax(scores)

    // 3. output = weighted sum of the values V_j.
    var out = [Double](repeating: 0.0, count: d)
    for j in 0...i {
        for k in 0..<d { out[k] += weights[j] * V[j][k] }
    }

    let w = weights.map { String(format: "%.2f", $0) }.joined(separator: ", ")
    let o = out.map { String(format: "%.2f", $0) }.joined(separator: ", ")
    print("token \(i + 1): weights=[\(w)]  →  out=[\(o)]")
}
