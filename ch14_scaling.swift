// Chapter 14. Scaling laws and the Chinchilla rule — in pure Swift.
//
// 1) The power law: the average loss predictably decreases as compute grows.
// 2) The Chinchilla rule: at a fixed budget, the optimum is ≈20 tokens per parameter.
//
// Run:
//     swift ch14_scaling.swift

import Foundation

// A power law of the form L(C) = E + a·C^(−b). E — the irreducible minimum (the entropy of language).
// The coefficients are toy ones, chosen for a realistic curve shape.
func loss(_ compute: Double) -> Double {
    let E = 1.6, a = 60.0, b = 0.0907
    return E + a * pow(compute, -b)
}

print("1) Scaling law: more compute → lower loss (predictably):")
for e in [18, 19, 20, 21, 22, 23, 24] {
    let C = pow(10.0, Double(e))
    print("  compute 10^\(e) FLOPs → loss ≈ \(String(format: "%.2f", loss(C)))")
}

// The Chinchilla rule: budget C ≈ 6·N·D, optimally D ≈ 20·N → C ≈ 120·N².
print("\n2) The Chinchilla rule (≈20 tokens per parameter) — how to split the budget:")
for flops in [1e19, 1e21, 1e23, 5.7e23, 1e25] {
    let N = (flops / 120.0).squareRoot()   // optimal number of parameters
    let D = 20.0 * N                         // optimal number of tokens
    print("  budget \(String(format: "%.1e", flops)) FLOPs → "
          + "params ≈ \(String(format: "%.1e", N)), tokens ≈ \(String(format: "%.1e", D))")
}
print("\n(5.7e23 ≈ the Chinchilla budget: ~70 billion parameters, ~1.4 trillion tokens — it matches!)")
