// Chapter 16. How many parameters are in our mini-GPT — in pure Swift (no MLX).
//
// We count the number of trainable weights from the hyperparameters, mirroring the
// structure of GPT.swift from the repository. It helps to get a feel for what makes up
// a model's size (and to check the ~12·d²·N formula from Chapter 11).
//
// Run:
//     swift ch16_paramcount.swift

import Foundation

// Default hyperparameters from Config.swift.
let nEmbd = 128
let nHead = 4
let nLayer = 4
let blockSize = 64
let vocabSize = 65          // roughly this many unique characters in the Shakespeare sample
let mlpHidden = 4 * nEmbd   // the classic ×4 FFN expansion

// Linear layer: inDim*outDim weights + (optionally) outDim biases.
func linear(_ inDim: Int, _ outDim: Int, bias: Bool) -> Int {
    inDim * outDim + (bias ? outDim : 0)
}
let layerNorm = 2 * nEmbd   // learnable scale (γ) and shift (β)

// One transformer block (as in GPT.swift):
let attn = 4 * linear(nEmbd, nEmbd, bias: false)   // Q, K, V and the output projection — no bias
let mlp = linear(nEmbd, mlpHidden, bias: true) + linear(mlpHidden, nEmbd, bias: true)
let block = layerNorm /*ln1*/ + attn + layerNorm /*ln2*/ + mlp

// The whole model:
let tokenEmb = vocabSize * nEmbd     // the token embedding table
let posEmb = blockSize * nEmbd       // the position embedding table
let head = linear(nEmbd, vocabSize, bias: false)   // the projection to logits
let total = tokenEmb + posEmb + nLayer * block + layerNorm /*lnFinal*/ + head

print("Parameters of the mini-GPT (nEmbd=\(nEmbd), nLayer=\(nLayer), nHead=\(nHead), vocab=\(vocabSize)):")
print("  token embeddings   : \(tokenEmb)")
print("  position embeddings: \(posEmb)")
print("  one block          : \(block)  × \(nLayer) layers = \(nLayer * block)")
print("  final LayerNorm    : \(layerNorm)")
print("  head (logits)      : \(head)")
print("  ─────────────")
print("  TOTAL: \(total)  (≈ \(String(format: "%.2f", Double(total) / 1e6)) million parameters)")
print("\nCompare: GPT-2 (small) — 124 million, GPT-3 — 175 billion. Ours is a toy, and that's good for learning.")
