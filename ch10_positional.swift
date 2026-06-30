// Chapter 10. Positional encoding (sinusoidal) — in pure Swift.
//
// We give each position in the text a unique signature vector made of sines and cosines
// of different frequencies. It's added to the word's embedding — and the model starts to
// "sense" word order. We show that the codes are unique and that similarity decays with distance.
//
// Run:
//     swift ch10_positional.swift

import Foundation

// PE(pos, 2i)   = sin(pos / 10000^(2i/d))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
func positionalEncoding(_ pos: Int, _ d: Int) -> [Double] {
    var pe = [Double](repeating: 0.0, count: d)
    for i in 0..<(d / 2) {
        let freq = 1.0 / pow(10000.0, Double(2 * i) / Double(d))
        pe[2 * i]     = sin(Double(pos) * freq)   // even dimensions — sine
        pe[2 * i + 1] = cos(Double(pos) * freq)   // odd ones — cosine
    }
    return pe
}

func cosSim(_ a: [Double], _ b: [Double]) -> Double {
    let dot = zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
    let na = sqrt(a.reduce(0.0) { $0 + $1 * $1 })
    let nb = sqrt(b.reduce(0.0) { $0 + $1 * $1 })
    return dot / (na * nb)
}

// 1. Uniqueness: print the codes of the first positions at a small d = 8 (so they fit).
print("Each position's code is unique (d = 8):")
for pos in 0..<6 {
    let pe = positionalEncoding(pos, 8).map { String(format: "%+.2f", $0) }
    print("  pos \(pos): [\(pe.joined(separator: ", "))]")
}

// 2. Closeness: at a realistic dimension d = 64, similarity smoothly decays with distance.
let d = 64
print("\nSimilarity of position 0 with the others (d = \(d)) — decays with distance:")
let p0 = positionalEncoding(0, d)
for pos in 1..<6 {
    let sim = cosSim(p0, positionalEncoding(pos, d))
    print("  pos 0 ↔ pos \(pos): \(String(format: "%.3f", sim))")
}
