from llm import answer_question
from retrieval import retrieve

def test_llm():
    print("===================================")
    print("Running LLM Test (LM Studio)")
    print("===================================\n")

    query = "What updates did Zoe say last meeting?"

    # Retrieve chunks
    chunks = retrieve(query, k=5)
    print("Retrieved Chunks:\n")
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print("Metadata:", c["metadata"])
        print(c["content"])
        print()

    # Ask the LLM
    print("\n===================================")
    print("LLM Answer")
    print("===================================\n")

    answer = answer_question(query, k=5)
    print(answer)

    print("\n===================================")
    print("Test complete.")
    print("===================================\n")


if __name__ == "__main__":
    test_llm()
