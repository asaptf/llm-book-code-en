// Chapter 5. Softmax, cross-entropy, and perplexity — in pure Swift.
//
// We show the whole "output" mechanism of a language model on one prediction step:
// logits -> softmax -> probabilities -> loss -> perplexity.
//
// Run:
//     swift ch05_softmax_loss.swift

import Foundation

// Logits — the "raw scores" a network would produce for four candidates.
// An array of pairs (not a dictionary), to keep the order on output.
let logits: [(String, Double)] = [
    ("mat", 2.1), ("sofa", 0.8), ("table", -0.5), ("window", 0.1),
]

/// Turns logits into probabilities. We subtract the maximum for stability.
func softmax(_ scores: [(String, Double)]) -> [(String, Double)] {
    let m = scores.map { $0.1 }.max()!
    let exps = scores.map { ($0.0, exp($0.1 - m)) }   // step 1: exponentiate
    let Z = exps.reduce(0.0) { $0 + $1.1 }            // normalizing sum
    return exps.map { ($0.0, $0.1 / Z) }              // step 2: divide by the sum
}

func f(_ x: Double, _ d: Int = 3) -> String { String(format: "%.\(d)f", x) }

let probs = softmax(logits)

print("Probabilities after softmax (sum = 1):")
for (w, p) in probs.sorted(by: { $0.1 > $1.1 }) {
    let bar = String(repeating: "█", count: Int((p * 30).rounded()))
    print("  \(w)\t\(f(p))  \(bar)")
}
let total = probs.reduce(0.0) { $0 + $1.1 }
print("  (sum check: \(f(total)))")

// In the training text the word "mat" actually stood there — that's the correct answer.
let trueToken = "mat"
let pTrue = probs.first { $0.0 == trueToken }!.1
let loss = -log(pTrue)          // single-step cross-entropy
let ppl = exp(loss)             // single-step perplexity

print("\nCorrect token: \(trueToken)")
print("Loss  -log(p): \(f(loss))")
print("Perplexity e^loss: \(f(ppl, 2))  (≈ between this many options the model wavers)")

let pTable = probs.first { $0.0 == "table" }!.1
print("\nIf 'table' were correct: loss \(f(-log(pTable))) — a much bigger penalty.")
