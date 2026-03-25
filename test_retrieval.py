from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Load vector DB
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VECTORDB_DIR = BASE_DIR / "vectordb"

client = chromadb.PersistentClient(path=str(VECTORDB_DIR))

print("📦 Available collections:", client.list_collections())

collection = client.get_collection("governance")

# ---------------------------------------------------------
# 2. Load embedding model
# ---------------------------------------------------------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ---------------------------------------------------------
# 3. Helper: run a query
# ---------------------------------------------------------
def test_query(query: str, filters: dict = None, k: int = 3):
    print("\n🔎 QUERY:", query)
    if filters:
        print("📌 Filters:", filters)

    query_embedding = model.encode(query).tolist()

    # Build query args safely (avoid empty where={})
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": k
    }
    if filters:
        query_args["where"] = filters

    results = collection.query(**query_args)

    # Handle empty results
    if not results["ids"] or len(results["ids"][0]) == 0:
        print("⚠️ No results found.")
        return

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    scores = results["distances"][0]

    for i in range(len(ids)):
        print("\n---------------- Result", i + 1, "----------------")
        print("ID:", ids[i])
        print("Score:", scores[i])
        print("Metadata:", metas[i])
        print("Content:\n", docs[i])
    print("--------------------------------------------\n")


# ---------------------------------------------------------
# 4. Run test cases
# ---------------------------------------------------------

print("\n===================================")
print("Running Retrieval Tests")
print("===================================\n")

# A. Minutes: role-specific retrieval
test_query(
    "What did VP Academic say about W&C?",
    filters={
        "$and": [
            {"document_type": "minutes"},
            {"role": "VP Academic"}
        ]
    }
)

# B. Minutes: sustainability discussion
test_query(
    "sustainability mandate restructuring",
    filters={"document_type": "minutes"}   # <-- FIXED
)

print("\n===================================")
print("Tests complete.")
print("===================================\n")
