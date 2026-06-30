// Chapter 13. The chat template — in pure Swift.
//
// After fine-tuning, the model expects input in a special format: turns are wrapped in
// special tokens (from Ch. 3) by which the model understands whose turn it is and where
// the answer ends. We show the popular ChatML format.
//
// Run:
//     swift ch13_chat_template.swift

import Foundation

struct Message {
    let role: String       // "system" | "user" | "assistant"
    let content: String
}

let chat = [
    Message(role: "system", content: "You are a polite assistant."),
    Message(role: "user", content: "Capital of France?"),
    Message(role: "assistant", content: "Paris."),
    Message(role: "user", content: "And Italy's?"),
]

// Glue the messages into a single string with the special tokens.
func applyChatTemplate(_ messages: [Message]) -> String {
    var s = ""
    for m in messages {
        s += "<|im_start|>\(m.role)\n\(m.content)<|im_end|>\n"
    }
    s += "<|im_start|>assistant\n"   // an invitation: "now it's your turn, assistant"
    return s
}

print("What the model actually receives (not just \"And Italy's?\"):")
print(String(repeating: "-", count: 50))
print(applyChatTemplate(chat), terminator: "")
print(String(repeating: "-", count: 50))
print("Next the model continues the text after the last <|im_start|>assistant —")
print("and we show that continuation to the user as the \"answer\".")
