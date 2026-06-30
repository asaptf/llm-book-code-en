// Chapter 4. Cosine similarity of embeddings — in pure Swift.
//
// No ML libraries: we set toy vectors by hand and see who resembles whom.
// The goal — to see that "closeness of meaning" = "closeness of vector directions".
//
// Run:
//     swift ch04_similarity.swift

import Foundation

// Toy 3-dimensional embeddings. In reality there are hundreds/thousands and they're learned.
// Here we "played the role of training" ourselves: animals have a large 1st coordinate,
// construction words a large 3rd.
let emb: [String: [Double]] = [
    "cat":      [0.9, 0.1, 0.0],
    "kitten":   [0.8, 0.2, 0.0],
    "dog":      [0.7, 0.1, 0.1],
    "puppy":    [0.6, 0.2, 0.1],
    "concrete": [0.0, 0.1, 0.9],
    "brick":    [0.1, 0.0, 0.8],
]

/// Cosine of the angle between vectors: a·b / (|a|·|b|). Range [-1, 1].
func cosSim(_ a: [Double], _ b: [Double]) -> Double {
    let dot = zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }   // dot product
    let na = sqrt(a.reduce(0.0) { $0 + $1 * $1 })          // length (norm) of vector a
    let nb = sqrt(b.reduce(0.0) { $0 + $1 * $1 })          // length (norm) of vector b
    return dot / (na * nb)
}

/// Finds the most similar word (other than itself).
func nearest(_ word: String) -> (String, Double) {
    emb.filter { $0.key != word }
       .map { ($0.key, cosSim(emb[word]!, $0.value)) }
       .max { $0.1 < $1.1 }!
}

func f(_ x: Double) -> String { String(format: "%.3f", x) }

print("Pairwise similarity (1.0 = same direction, 0 = unrelated):")
print("  cat ↔ kitten    : \(f(cosSim(emb["cat"]!, emb["kitten"]!)))")
print("  cat ↔ dog       : \(f(cosSim(emb["cat"]!, emb["dog"]!)))")
print("  cat ↔ concrete  : \(f(cosSim(emb["cat"]!, emb["concrete"]!)))")
print("  concrete ↔ brick: \(f(cosSim(emb["concrete"]!, emb["brick"]!)))")
print("\nNearest neighbor of each word:")
for w in emb.keys.sorted() {
    let (best, score) = nearest(w)
    print("  \(w) → \(best) (\(f(score)))")
}
