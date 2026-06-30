// Chapter 12. Why pretraining is "unsupervised": the text is its own labeling.
//
// From ordinary text, with no manual labeling at all, you get a sea of training
// examples of "context → next token". We show this on characters (as in Ch. 3).
//
// Run:
//     swift ch12_training_pairs.swift

import Foundation

let text = "token"
let chars = Array(text)

print("From the text \"\(text)\" we get training pairs (context → correct answer):")
var count = 0
for i in 1..<chars.count {
    let context = String(chars[0..<i])   // everything to the left
    let target = chars[i]                // the next character is the "answer"
    print("  \"\(context)\"\t→  '\(target)'")
    count += 1
}
print("\nOne word of \(chars.count) characters gave \(count) training examples.")
print("No one labeled the data by hand — the \"correct answer\" was already in the text.")

// Scale: the number of examples ≈ the number of tokens in the corpus.
let corpusTokens = 1_000_000_000_000   // ~a trillion tokens (the order of a large model)
print("\nA corpus of \(corpusTokens) tokens → roughly that many training examples.")
print("This is why \"just text\" is enough: there are trillions of tokens of it on the internet.")
