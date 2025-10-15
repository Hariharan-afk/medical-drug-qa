import sys

from intent_classifier import classify_intent
from retriever import retrieve
from answer_generator import generate_answer


def chat_loop():
    print("Medical QA Chatbot (type 'exit' or 'quit' to stop)")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query or query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # 1) Intent classification
        intent = classify_intent(query)
        drug = intent.get("drug_name")
        section = intent.get("section")

        if not drug:
            print("Sorry, I don’t have information on that drug.")
            continue

        # 2) Retrieve top chunks
        chunks = retrieve(query, drug, section)
        if not chunks:
            print(f"Sorry, I don’t have any '{section}' info for {drug}.")
            continue

        # 3) Generate final answer
        try:
            answer = generate_answer(query, chunks)
        except Exception as e:
            print(f"Error generating answer: {e}")
            continue

        print(f"\n{answer}")


if __name__ == '__main__':
    # allow passing a single question on the command line
    if len(sys.argv) > 1:
        single_q = " ".join(sys.argv[1:])
        intent = classify_intent(single_q)
        drug = intent.get("drug_name")
        section = intent.get("section")
        if not drug:
            print("No matching drug in database.")
            sys.exit(1)
        chunks = retrieve(single_q, drug, section)
        if not chunks:
            print(f"No '{section}' info for {drug}.")
            sys.exit(1)
        print(generate_answer(single_q, chunks))
    else:
        chat_loop()

