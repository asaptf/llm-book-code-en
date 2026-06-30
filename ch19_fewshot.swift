// Chapter 19. Assembling a few-shot prompt — in pure Swift.
//
// "In-context learning" (few-shot): we do NOT fine-tune the model, we just put
// a few "input → answer" examples in the prompt. The model continues the pattern on a
// new input. We show how such a prompt is assembled from parts.
//
// Run:
//     swift ch19_fewshot.swift

import Foundation

struct Example {
    let input: String
    let output: String
}

let task = "Determine the sentiment of the review: positive or negative."
let shots = [
    Example(input: "Great movie, highly recommend!", output: "positive"),
    Example(input: "Boring, left halfway through.", output: "negative"),
    Example(input: "Best purchase of the year.", output: "positive"),
]
let query = "Waste of money, don't buy it."

func buildFewShot(task: String, shots: [Example], query: String) -> String {
    var p = task + "\n\n"
    for s in shots {
        p += "Review: \(s.input)\nSentiment: \(s.output)\n\n"   // the sample examples
    }
    p += "Review: \(query)\nSentiment:"   // the model will continue RIGHT here
    return p
}

print("The prompt that actually goes to the model:")
print(String(repeating: "-", count: 52))
print(buildFewShot(task: task, shots: shots, query: query))
print(" ⟵ here the model will write \"negative\", following the pattern")
print(String(repeating: "-", count: 52))
print("No fine-tuning: the examples in the prompt set a template, and next-token")
print("prediction naturally continues it with the correct answer.")
