// Chapter 15. The data and tokenizer logic from mini-gpt — without MLX, so it runs.
//
// This is a simplified, self-contained version of Corpus.swift from the companion repo:
// vocabulary → encoding → train/val split → batch (x, y), where y is x shifted by one
// position (for each position, y holds "the next character").
// The production version (with MLX tensors) is mini-gpt/Sources/MiniGPT/Corpus.swift.
//
// Run:
//     swift ch15_corpus_demo.swift

import Foundation

let text = "hello, world! hello, GPT!"

// 1. Vocabulary = unique characters, sorted (determinism between runs).
let itos = Array(Set(text)).sorted()
var stoi = [Character: Int]()
for (i, c) in itos.enumerated() { stoi[c] = i }

func encode(_ s: String) -> [Int] { s.compactMap { stoi[$0] } }
func decode(_ ids: [Int]) -> String { String(ids.map { itos[$0] }) }

let data = encode(text)
print("Text: \(text)")
print("Vocabulary size: \(itos.count) characters")
print("Encoded into \(data.count) tokens")

// 2. train/val split — the first 90% for training, the tail for validation.
let splitPoint = Int(Double(data.count) * 0.9)
let trainData = Array(data[..<splitPoint])
let valData = Array(data[splitPoint...])
print("train/val: \(trainData.count) / \(valData.count)")

// 3. One training example: x = a window, y = the same window, shifted by 1 character.
let blockSize = 8
let i = 0   // fix the start for reproducible output (in training it's random)
let x = Array(trainData[i ..< i + blockSize])
let y = Array(trainData[(i + 1) ..< (i + 1 + blockSize)])
print("\nx (context): \(x)  → \"\(decode(x))\"")
print("y (target):  \(y)  → \"\(decode(y))\"")

// One window of blockSize characters = blockSize examples of "context → next character":
print("\nWhat we learn from this window:")
for t in 0 ..< blockSize {
    let ctx = decode(Array(x[0 ... t]))
    let next = decode([y[t]])
    print("  \"\(ctx)\" → '\(next)'")
}
