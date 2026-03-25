import sys
from llm import answer_question

DOCUMENT_TYPE = None
ROLE          = None
K = 5

def ask(query: str) -> str:
    return answer_question(
        query,
        k=K,
        document_type=DOCUMENT_TYPE,
        role=ROLE
    )


def interactive_loop():
    print("=" * 50)
    print("  ESS Governance Assistant")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        print("\nAnswer:")
        print(ask(query))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(ask(query))
    else:
        interactive_loop()