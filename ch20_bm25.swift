// Chapter 20. BM25 search — the core of RAG, in pure Swift (no MLX/LLM).
//
// A simplified version of BM25.swift from swift-rag: we build an inverted index over
// text chunks and rank them for a query with the classic BM25 formula. This is "retrieval"
// (search) — the first half of RAG. Fully deterministic.
//
// Run:
//     swift ch20_bm25.swift

import Foundation

// A "corpus" of short chunks on different topics.
let chunks = [
    "A bicycle is a two-wheeled vehicle propelled by pedals.",
    "Bees collect nectar from flowers and produce honey in their hives.",
    "A volcano erupts molten lava and ash from the earth's crust.",
    "A mountain bike has wide tires and shock absorbers for off-road.",
]

let stopwords: Set<String> = ["a", "an", "the", "is", "and", "in", "on", "with", "from",
                              "for", "by", "to", "of", "their", "its", "how", "do", "this"]

// Analyzer: lowercase, split into words, drop short ones and stopwords.
func analyze(_ text: String) -> [String] {
    var terms: [String] = [], cur = ""
    func emit() {
        defer { cur = "" }
        if cur.count >= 2 && !stopwords.contains(cur) { terms.append(cur) }
    }
    for ch in text.lowercased() {
        if ch.isLetter || ch.isNumber { cur.append(ch) } else { emit() }
    }
    emit()
    return terms
}

// Build the inverted index: term -> [(chunk number, how many times it occurred)].
var postings: [String: [(doc: Int, tf: Int)]] = [:]
var docLen = [Int]()
for (i, c) in chunks.enumerated() {
    let terms = analyze(c)
    docLen.append(terms.count)
    var tf = [String: Int]()
    for t in terms { tf[t, default: 0] += 1 }
    for (t, n) in tf { postings[t, default: []].append((i, n)) }
}
let avgLen = Double(docLen.reduce(0, +)) / Double(chunks.count)

// BM25 search.
func search(_ query: String, k1: Double = 1.5, b: Double = 0.75) -> [(doc: Int, score: Double)] {
    let N = Double(chunks.count)
    var scores = [Double](repeating: 0, count: chunks.count)
    for term in Set(analyze(query)) {
        guard let list = postings[term] else { continue }
        let df = Double(list.count)
        let idf = log(1.0 + (N - df + 0.5) / (df + 0.5))            // rarer word is more important
        for p in list {
            let tf = Double(p.tf), dl = Double(docLen[p.doc])
            let denom = tf + k1 * (1 - b + b * dl / avgLen)         // length normalization
            scores[p.doc] += idf * (tf * (k1 + 1)) / denom
        }
    }
    return scores.enumerated().filter { $0.element > 0 }
        .map { (doc: $0.offset, score: $0.element) }
        .sorted { $0.score > $1.score }
}

let query = "how do bees make honey"
print("Query: \"\(query)\"")
print("Relevant query words: \(Set(analyze(query)).sorted())")
print("\nChunks ranked by BM25:")
for hit in search(query) {
    print("  [\(String(format: "%.3f", hit.score))]  \(chunks[hit.doc])")
}
