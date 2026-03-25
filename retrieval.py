from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Load vector DB
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VECTORDB_DIR = BASE_DIR / "vectordb"

client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
collection = client.get_collection("governance")

# ---------------------------------------------------------
# 2. Load embedding model
# ---------------------------------------------------------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ---------------------------------------------------------
# 3. Meeting index + date resolution
# ---------------------------------------------------------

def get_meeting_index() -> list[dict]:
    results = collection.get(include=["metadatas"])

    seen = {}
    for meta in results["metadatas"]:
        iso = meta.get("meeting_date", "")
        display = meta.get("meeting_date_display", "")
        if iso and iso not in seen:
            seen[iso] = display

    return [
        {"iso": iso, "display": display}
        for iso, display in sorted(seen.items(), reverse=True)
    ]

def resolve_meeting_date(query: str) -> str | None:
    """
    Inspect the query for temporal references and return the matching
    ISO date string to use as a ChromaDB filter, or None if no specific
    meeting is referenced (meaning: don't filter, search all meetings).
    """
    meetings = get_meeting_index()
    if not meetings:
        return None

    query_lower = query.lower()

    # Relative: most recent
    if any(p in query_lower for p in ["last meeting", "most recent", "latest meeting"]):
        return meetings[0]["iso"]

    # Relative: second most recent
    if any(p in query_lower for p in ["previous meeting", "meeting before last"]):
        return meetings[1]["iso"] if len(meetings) > 1 else meetings[0]["iso"]

    # Explicit date mention: match against known display strings and ISO strings
    # e.g. "february 23", "feb 23", "2026-02-23"
    for m in meetings:
        display = m["display"].lower()          # "february 23, 2026"
        iso = m["iso"]                          # "2026-02-23"
        month_day = " ".join(display.split()[:2]).rstrip(",")  # "february 23"
        if month_day in query_lower or iso in query_lower:
            return iso

    return None  # No meeting reference found — do not filter


# ---------------------------------------------------------
# 4. Build valid Chroma filters
# ---------------------------------------------------------
def build_filters(document_type=None, role=None, meeting_date=None):
    clauses = []

    if document_type:
        clauses.append({"document_type": document_type})

    if role:
        clauses.append({"role": role})

    if meeting_date:
        clauses.append({"meeting_date": meeting_date})

    if len(clauses) == 0:
        return None
    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


# ---------------------------------------------------------
# 5. Retrieval function
# ---------------------------------------------------------
def retrieve(query: str, k: int = 5, document_type: str = None, role: str = None, meeting_date: str = None) -> list:
    filters = build_filters(document_type, role, meeting_date)

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    query_embedding = model.encode(BGE_QUERY_PREFIX + query).tolist()

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": k
    }

    if filters:
        query_args["where"] = filters

    results = collection.query(**query_args)

    if not results["ids"] or len(results["ids"][0]) == 0:
        return []

    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id": results["ids"][0][i],
            "score": results["distances"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i]
        })

    return output