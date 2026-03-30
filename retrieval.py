from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import re

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

def get_meeting_index(doc_subtype_filter: str = None) -> list[str]:
    """Return a sorted list of known meeting ISO dates, most recent first."""
    # include=["metadatas"] is correct, but let's ensure we filter strictly
    results = collection.get(include=["metadatas"])

    seen = set()
    for meta in results["metadatas"]:
        # CRITICAL: If the user wants BoD, only look at BoD dates
        if doc_subtype_filter:
            if doc_subtype_filter not in meta.get("document_subtype", "").lower():
                continue

        iso = meta.get("meeting_date", "")
        if iso and iso != "unknown":
            seen.add(iso)

    return sorted(seen, reverse=True)


def detect_document_type(query: str) -> str | None:
    """Detect if query is about policy, bylaws, or meeting types. Returns document type or None."""
    lower = query.lower()

    # Policy/bylaws keywords are the most specific indicators.
    if "policy" in lower or "procedure" in lower:
        return "policy"
    if "bylaw" in lower or "by-law" in lower or "constitution" in lower:
        return "bylaws"

    # Implicit policy patterns: questions about governance rules, officer duties, organizational procedures
    policy_patterns = [
        # Officer actions and responsibilities
        ("officer" in lower and ("resign" in lower or "step down" in lower or "leave" in lower or "election" in lower or "elected" in lower or "responsib" in lower or "duties" in lower or "position" in lower)),
        # Authority/permissions/approvals
        "approve" in lower,
        "authority" in lower,
        "permission" in lower,
        "allowed" in lower,
        "can spend" in lower,
        # Governance structure and roles
        ("executive" in lower or "committee" in lower) and ("role" in lower or "position" in lower or "member" in lower),
        # Dues, fees, membership rules
        "dues" in lower,
        "fee" in lower,
        "membership" in lower,
        # Meeting requirements and quorum
        "quorum" in lower,
        ("meeting" in lower and "required" in lower) or "meeting quorum" in lower,
        # Voting and procedures
        "motion" in lower and ("pass" in lower or "seconded" in lower or "carried" in lower),
        # What happens if/then (procedural)
        "what happens if" in lower and not ("meeting" in lower or "minutes" in lower),
        "what should" in lower and ("do" in lower or "happen" in lower),
        "how do" in lower and ("handle" in lower or "deal with" in lower or "address" in lower),
    ]

    if any(pattern for pattern in policy_patterns if pattern):
        return "policy"

    return None


def resolve_meeting_date(query: str) -> tuple[str | None, str | None]:
    lower = query.lower()
    doc_subtype = None

    # Identify which "lane" we are in
    if "bod" in lower or "board" in lower:
        doc_subtype = "bod"
    elif "officer" in lower:
        doc_subtype = "officer"

    # Get dates ONLY for that specific type
    dates = get_meeting_index(doc_subtype)

    if ("last" in lower or "recent" in lower) and dates:
        # This now returns the last BOD date if doc_subtype is bod
        return dates[0], doc_subtype

    return None, doc_subtype

# ---------------------------------------------------------
# 4. Build valid Chroma filters
# ---------------------------------------------------------
def build_filters(document_type=None, document_subtype=None, role=None, meeting_date=None):
    clauses = []

    if document_type:
        clauses.append({"document_type": document_type})

    if document_subtype:
        clauses.append({"document_subtype": document_subtype})

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
def retrieve(query: str, k: int = 20, document_type: str = None, document_subtype: str = None, role: str = None, meeting_date: str = None) -> list:
    filters = build_filters(document_type, document_subtype, role, meeting_date)

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