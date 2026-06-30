// Chapter 18. Sampling strategies: temperature, top-k, top-p — in pure Swift.
//
// We take a fixed distribution of logits and see how different "knobs" change the
// probabilities of choosing the next token. This is the math inside
// sampleNextCharacter from Training.swift (companion repository).
//
// Run:
//     swift ch18_sampling.swift

import Foundation

let vocab = ["the", "a", "cat", "dog", "quantum"]
let logits: [Double] = [2.0, 1.5, 1.0, 0.5, -1.0]   // "raw scores" from the model

// softmax with temperature: the logits are divided by t before the exponential.
func softmax(_ xs: [Double], temperature t: Double = 1.0) -> [Double] {
    let scaled = xs.map { $0 / t }
    let m = scaled.max()!
    let e = scaled.map { exp($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}

func show(_ title: String, _ probs: [Double]) {
    print(title)
    for (w, p) in zip(vocab, probs) {
        let name = w.padding(toLength: 8, withPad: " ", startingAt: 0)
        let bar = String(repeating: "█", count: Int((p * 40).rounded()))
        print("  \(name) \(String(format: "%.3f", p)) \(bar)")
    }
    print("")
}

// 1. Temperature: <1 sharpens (more confident), >1 flattens (riskier).
show("temperature = 1.0 (as is):", softmax(logits))
show("temperature = 0.5 (cautious):", softmax(logits, temperature: 0.5))
show("temperature = 1.5 (bold):", softmax(logits, temperature: 1.5))

// 2. top-k: keep the k most probable, zero out the rest, and renormalize.
func topK(_ probs: [Double], _ k: Int) -> [Double] {
    let threshold = probs.sorted(by: >)[k - 1]   // the k-th largest probability
    let kept = probs.map { $0 >= threshold ? $0 : 0.0 }
    let s = kept.reduce(0, +)
    return kept.map { $0 / s }
}
show("top-k = 2 (only the 2 best):", topK(softmax(logits), 2))

// 3. top-p (nucleus): keep the smallest set with combined probability >= p.
func topP(_ probs: [Double], _ p: Double) -> [Double] {
    let order = probs.indices.sorted { probs[$0] > probs[$1] }   // indices in descending order
    var cumulative = 0.0
    var keep = Set<Int>()
    for i in order {
        keep.insert(i)
        cumulative += probs[i]
        if cumulative >= p { break }   // once we've accumulated p — stop
    }
    let kept = probs.indices.map { keep.contains($0) ? probs[$0] : 0.0 }
    let s = kept.reduce(0, +)
    return kept.map { $0 / s }
}
show("top-p = 0.9 (90% nucleus):", topP(softmax(logits), 0.9))
