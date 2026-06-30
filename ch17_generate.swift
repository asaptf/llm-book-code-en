// Chapter 17. Autoregressive generation — a toy one, in pure Swift (no MLX).
//
// The real generate() from Training.swift runs the trained model. To show the
// generation LOOP itself runnably, we replace the model with the simplest bigram
// statistic: from the last character we take the distribution of the next from the
// text's frequencies, sample with temperature, append, repeat. The loop is exactly
// like the one in generate().
//
// Run:
//     swift ch17_generate.swift

import Foundation

// A seeded generator (SplitMix64), so the output is reproducible between runs.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
var rng = SplitMix64(seed: 7)

let text = """
to be or not to be that is the question
whether tis nobler in the mind to suffer
"""

// Bigrams: for each character — counts of which character came next.
var counts: [Character: [Character: Int]] = [:]
let chars = Array(text)
for i in 0 ..< (chars.count - 1) {
    counts[chars[i], default: [:]][chars[i + 1], default: 0] += 1
}

// Sample the next character: logits = log(count), divide by temperature,
// softmax, draw from the categorical distribution. (The same as sampleNextCharacter.)
func sampleNext(after c: Character, temperature: Double) -> Character? {
    guard let next = counts[c], !next.isEmpty else { return nil }
    // Sort by character: otherwise the dictionary's iteration order is random between runs
    // (Swift randomizes the hash), and the output would not be reproducible.
    let items = next.sorted { $0.key < $1.key }
    let scores = items.map { log(Double($0.value)) / temperature }
    let m = scores.max()!
    var probs = scores.map { exp($0 - m) }
    let sum = probs.reduce(0, +)
    probs = probs.map { $0 / sum }
    let r = Double.random(in: 0 ..< 1, using: &rng)
    var cumulative = 0.0
    for (idx, p) in probs.enumerated() {
        cumulative += p
        if r < cumulative { return items[idx].key }
    }
    return items.last?.key
}

// The autoregression loop: append one character at a time.
func generate(start: Character, length: Int, temperature: Double) -> String {
    var out = [start], cur = start
    for _ in 0 ..< length {
        guard let nxt = sampleNext(after: cur, temperature: temperature) else { break }
        out.append(nxt)
        cur = nxt
    }
    return String(out)
}

print("The same loop as in generate(): sample → append → repeat.")
print("temp 0.5 (cautious): \(generate(start: "t", length: 50, temperature: 0.5))")
print("temp 1.0 (as is):    \(generate(start: "t", length: 50, temperature: 1.0))")
print("\n(Bigrams see only 1 character back — hence the incoherence. The transformer sees the whole context.)")
