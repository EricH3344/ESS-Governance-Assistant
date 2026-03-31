from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import re

# Load vector DB
BASE_DIR = Path(__file__).resolve().parent
VECTORDB_DIR = BASE_DIR / "vectordb"

client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
collection = client.get_or_create_collection("governance")

# Load embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ---------------------------------------------------------
# Utility functions for retrieval
#
# Function: detect document type and meeting date from query, 
# build ChromaDB filters, and perform retrieval
# 
# Returns: a list of retrieved chunks with metadata 
# based on the query and detected context
# ---------------------------------------------------------
def get_meeting_index(doc_subtype_filter: str = None) -> list[str]:
    # Retrieve all meeting dates from the collection
    results = collection.get(include=["metadatas"])

    seen = set()
    for meta in results["metadatas"]:
        if meta.get("document_type", "").lower() != "minutes":
            continue

        # Apply document subtype filter if specified
        if doc_subtype_filter:
            subtype = (meta.get("document_subtype") or "").lower()
            source = (meta.get("source") or "").lower()
            if doc_subtype_filter == "bod":
                if "bod" not in subtype and "bod" not in source and "board" not in source and "board" not in subtype:
                    continue
            elif doc_subtype_filter == "officer":
                if "officer" not in subtype and "officer" not in source and "executive" not in subtype and "executive" not in source:
                    continue
            elif doc_subtype_filter not in subtype and doc_subtype_filter not in source:
                continue

        iso = meta.get("meeting_date", "")
        if iso and iso != "unknown":
            seen.add(iso)

    return sorted(seen, reverse=True)

# ---------------------------------------------------------
# Detect document type
#
# Function: analyze the query to determine if it's asking 
# about policies, bylaws, or meetings
# 
# Returns: a string indicating the detected document type 
# or None if not clear
# ---------------------------------------------------------
def detect_document_type(query: str) -> str | None:
    lower = query.lower()

    if "policy" in lower or "procedure" in lower:
        return "policy"
    if "bylaw" in lower or "by-law" in lower or "constitution" in lower:
        return "bylaws"

    return None

# ---------------------------------------------------------
# Resolve meeting date from query
#
# Function: analyze the query for references to meetings 
# and use the meeting index to find the most relevant date
# 
# Returns: a tuple of meeting_date and document_subtype
# ---------------------------------------------------------
def resolve_meeting_date(query: str) -> tuple[str | None, str | None]:
    lower = query.lower()
    doc_subtype = None

    if "bod" in lower or "board" in lower:
        doc_subtype = "bod"
    elif "officer" in lower:
        doc_subtype = "officer"

    dates = get_meeting_index(doc_subtype)

    # Helper function for common probable prompts
    if ("last" in lower or "recent" in lower) and dates:
        return dates[0], doc_subtype

    return None, doc_subtype

# ---------------------------------------------------------
# Build ChromaDB filters
#
# Function: build a filter dictionary for ChromaDB queries 
# based on the detected document type, subtype, role, and meeting date
# 
# Returns: a filter dictionary to be used in ChromaDB queries
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
# Retrieval function
#
# Function: perform a retrieval from ChromaDB based on the 
# query and detected context, using the appropriate filters
# 
# Returns: a list of retrieved documents
# ---------------------------------------------------------
def retrieve(query: str, k: int = 20, document_type: str = None, document_subtype: str = None, role: str = None, meeting_date: str = None) -> list:
    filters = build_filters(document_type, document_subtype, role, meeting_date)

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    query_embedding = model.encode(BGE_QUERY_PREFIX + query).tolist()

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": k
    }

    # Query filtered documents
    if filters:
        query_args["where"] = filters

    results = collection.query(**query_args)

    # If no results are found with the document subtype filter, try without
    if (not results["ids"] or len(results["ids"][0]) == 0) and document_subtype:
        fallback_filters = build_filters(document_type=document_type, document_subtype=None, role=role, meeting_date=meeting_date)
        if fallback_filters is not None:
            query_args["where"] = fallback_filters
            results = collection.query(**query_args)

    if not results["ids"] or len(results["ids"][0]) == 0:
        return []

    # Format results into a list of dictionaries with id, score, content, and metadata
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "id": results["ids"][0][i],
            "score": results["distances"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i]
        })

    return output