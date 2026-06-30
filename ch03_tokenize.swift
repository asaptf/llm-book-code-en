// Chapter 3. Tokenization from scratch — in pure Swift.
//
// The simplest tokenizer: character-level. We build a vocabulary of the text's
// unique characters and learn to encode text into numbers and back. Real models
// add BPE merges on top (see the companion repo, mini-gpt/Sources/MiniGPT/Corpus.swift),
// but the idea "text ↔ list of numbers" is exactly this.
//
// Run:
//     swift ch03_tokenize.swift

import Foundation

let text = "Tokenization is simple."

// 1. Vocabulary: the text's unique characters, sorted for reproducibility.
let chars = Array(Set(text)).sorted()

// 2. Two lookup tables: character -> id and id -> character.
var stoi: [Character: Int] = [:]   // string-to-index
var itos: [Int: Character] = [:]   // index-to-string
for (i, c) in chars.enumerated() {
    stoi[c] = i
    itos[i] = c
}

// 3. encode: text -> list of token ids. decode: back again.
func encode(_ s: String) -> [Int] { s.map { stoi[$0]! } }
func decode(_ ids: [Int]) -> String { String(ids.map { itos[$0]! }) }

let ids = encode(text)

print("Original text: \(text)")
print("Vocabulary size: \(chars.count) characters")
print("Characters in text: \(text.count), tokens: \(ids.count)")
print("Tokens (id): \(ids)")
print(String(repeating: "-", count: 40))

// Show the mapping of the first characters to their ids.
for c in chars.prefix(8) {
    print("  '\(c)' → \(stoi[c]!)")
}

print(String(repeating: "-", count: 40))
// Check reversibility: list of ids -> original text without loss.
let restored = decode(ids)
print("Restored: \(restored)")
assert(restored == text, "Tokenization must be reversible!")
print("Reversibility confirmed ✓")
