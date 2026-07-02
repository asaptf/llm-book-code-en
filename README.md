# How LLMs Work — Book Source Code

Source code for the book **"How LLMs Work: From Zero to Your Own Transformer"** by Andrey Sapunov — the **English edition**.

📚 **Get the book on Amazon:** [amazon.com/dp/B0H7BPJLHC](https://www.amazon.com/dp/B0H7BPJLHC)

This repository contains the small, self-contained code listings from the book — runnable Swift examples that illustrate each chapter's ideas. The full mini-GPT model and the RAG project live in a separate repository: [github.com/asaptf/swift-language-models](https://github.com/asaptf/swift-language-models). The Russian-edition listings are at [github.com/asaptf/llm-book-code-ru](https://github.com/asaptf/llm-book-code-ru).

---

## What's inside

Each file corresponds to a chapter of the book and illustrates one idea. Every example is a **self-contained pure-Swift file with no dependencies** — no MLX, no GPU, no extra packages required. The aim is clarity, not performance: you can read each file top to bottom and run it in one line.

| File | Chapter | What it shows |
|------|---------|---------------|
| [`ch03_tokenize.swift`](ch03_tokenize.swift) | 3 | Tokenization from scratch (character-level) |
| [`ch04_similarity.swift`](ch04_similarity.swift) | 4 | Cosine similarity of embeddings |
| [`ch05_softmax_loss.swift`](ch05_softmax_loss.swift) | 5 | Softmax, cross-entropy, and perplexity |
| [`ch06_forward.swift`](ch06_forward.swift) | 6 | The forward pass of a tiny neural network |
| [`ch07_gradient_descent.swift`](ch07_gradient_descent.swift) | 7 | Gradient descent, live |
| [`ch08_attention.swift`](ch08_attention.swift) | 8 | Self-attention with a causal mask |
| [`ch09_ffn.swift`](ch09_ffn.swift) | 9 | The transformer's feed-forward network (FFN) |
| [`ch10_positional.swift`](ch10_positional.swift) | 10 | Sinusoidal positional encoding |
| [`ch11_block.swift`](ch11_block.swift) | 11 | A full transformer block (pre-LN + residuals) |
| [`ch12_training_pairs.swift`](ch12_training_pairs.swift) | 12 | Why pretraining is "unsupervised" |
| [`ch13_chat_template.swift`](ch13_chat_template.swift) | 13 | The chat template |
| [`ch14_scaling.swift`](ch14_scaling.swift) | 14 | Scaling laws and the Chinchilla rule |
| [`ch15_corpus_demo.swift`](ch15_corpus_demo.swift) | 15 | The data & tokenizer logic from mini-gpt |
| [`ch16_paramcount.swift`](ch16_paramcount.swift) | 16 | Counting the parameters of a mini-GPT |
| [`ch17_generate.swift`](ch17_generate.swift) | 17 | Autoregressive generation (toy bigram) |
| [`ch18_sampling.swift`](ch18_sampling.swift) | 18 | Sampling: temperature, top-k, top-p |
| [`ch19_fewshot.swift`](ch19_fewshot.swift) | 19 | Assembling a few-shot prompt |
| [`ch20_bm25.swift`](ch20_bm25.swift) | 20 | BM25 search — the core of RAG |
| [`ch22_quantization.swift`](ch22_quantization.swift) | 22 | Quantization and the memory cost of weights |

---

## Requirements

Just the **Swift compiler** — nothing else. These examples are cross-platform and run anywhere Swift is installed (macOS, Linux, Windows). They need neither MLX nor a graphics card.

**macOS:**

```bash
xcode-select --install      # once: installs the Command Line Tools (incl. swift)
swift --version             # check
```

**Linux / Windows:** install Swift from [swift.org/install](https://www.swift.org/install/).

---

## Running an example

Every file runs in a single line:

```bash
swift ch05_softmax_loss.swift
```

Each file begins with a comment explaining what it does, and most print a short, readable trace of the computation so you can see the idea in action. Several examples use a seeded random generator, so their output is reproducible between runs.

---

## Going further

- **The full mini-GPT** (Part V of the book) is built on Apple's **MLX** framework and requires a Mac on Apple Silicon. It lives in [swift-language-models](https://github.com/asaptf/swift-language-models) under `mini-gpt/`.
- **The RAG project** (Chapter 20) combines the BM25 search you see in `ch20_bm25.swift` with a small local model; it's in the same repository under `swift-rag/`.

See **Appendix D** of the book for the full setup guide.

---

## About the book

The book explains how large language models actually work — from "what even is this thing?" all the way to the line of code where your own small model generates its first coherent text. The main text is written in plain language with analogies and pictures; engineering asides go deeper with formulas and code. The source code in this repository is those asides made runnable.

**[Buy on Amazon →](https://www.amazon.com/dp/B0H7BPJLHC)**

*Code examples in **Swift** (pure Swift and MLX). Edition 1.0.*

## License

These code examples are provided for readers of the book to learn from and experiment with.