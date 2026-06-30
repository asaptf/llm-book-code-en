// Chapter 22. Quantization: how much memory the weights take at different precisions.
//
// One key to the future is running LLMs "on the device." The main technique is
// quantization: storing weights not in 32/16 bits but in 8 or 4. Let's compute the savings.
// Pure arithmetic, no MLX.
//
// Run:
//     swift ch22_quantization.swift

import Foundation

let params = 7_000_000_000   // 7 billion parameters — a typical "small" open model

let formats: [(name: String, bits: Int)] = [
    ("fp32 (full precision) ", 32),
    ("fp16 (typical for GPU)", 16),
    ("int8 (quantized)      ", 8),
    ("int4 (aggressive)     ", 4),
]

let gib = 1_073_741_824.0   // bytes in one gigabyte (2^30)
print("Memory for the weights of a \(params / 1_000_000_000)-billion-parameter model:")
for f in formats {
    let bytes = Double(params) * Double(f.bits) / 8.0   // bits → bytes
    print("  \(f.name): \(String(format: "%5.1f", bytes / gib)) GB")
}

print("""

int4 shrinks the model 8-fold versus fp32 — and a 7B model fits into a laptop's memory.
Quality dips slightly, but usually tolerably. This is the road to LLMs "on the device"
(Apple Silicon + MLX, llama.cpp, etc. — see "Why Swift" in the Introduction).
""")
