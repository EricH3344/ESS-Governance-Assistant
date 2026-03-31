from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "Processed"
VECTORDB_DIR = BASE_DIR / "vectordb"

PROCESSED_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)

# Load vector DB
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Initialize ChromaDB client and collection
chroma = chromadb.PersistentClient(path=str(VECTORDB_DIR))
collection = chroma.get_or_create_collection(
    name="governance",
    # Use cosine distance for better semantic search performance
    metadata={"hnsw:space": "cosine"}
)

# ---------------------------------------------------------
# Build embedding text
#
# Function: create a text string for embedding that combines 
# the content and key metadata fields in a structured format.
# 
# Returns: a string that includes the content and metadata for embedding
# ---------------------------------------------------------
def build_embedding_text(chunk: dict) -> str:
    text = (
        f"[document_type: {chunk.get('document_type','unknown')}]\n"
        f"[type: {chunk.get('type','unknown')}]\n"
        f"[role: {chunk.get('role','N/A')}]\n"
        f"[person: {chunk.get('person','N/A')}]\n"
        f"{chunk.get('content', '')}"
    )
    return text

# Process each JSON file in the processed directory and upsert into ChromaDB
for json_file in PROCESSED_DIR.glob("*.json"):
    print(f"Loading: {json_file.name}")
    chunks = json.loads(json_file.read_text(encoding="utf-8"))

    for i, chunk in enumerate(chunks):
        source = chunk.get("source_file", json_file.name)
        chunk_id = f"{source}_{i}"

        text_for_embedding = build_embedding_text(chunk)
        embedding = model.encode("Represent this sentence for searching relevant passages: " + text_for_embedding).tolist()

        document_subtype = chunk.get("document_subtype")
        if not document_subtype:
            document_subtype = "unknown"

        if document_subtype == "unknown" and chunk.get("document_type") == "minutes":
            filename = source.lower()
            if "officer" in filename or "executive" in filename:
                document_subtype = "officer"
            elif "bod" in filename or "board" in filename:
                document_subtype = "bod"

        metadata = {
            "source": source,
            "document_type": chunk.get("document_type", "unknown"),
            "type": chunk.get("type", "unknown"),
            "role": chunk.get("role", "N/A"),
            "person": chunk.get("person", "N/A"),
            "chunk_index": i,
            "meeting_date": chunk.get("meeting_date", ""),
            "meeting_date_display": chunk.get("meeting_date_display", ""), #
        }

        # Only store meeting_date for minutes chunks that actually have a date
        if chunk.get("meeting_date"):
            metadata["meeting_date"] = chunk["meeting_date"]
            metadata["meeting_date_display"] = chunk.get("meeting_date_display", "")

        # Policy/bylaws extras
        if chunk.get("type") == "policy_section" or chunk.get("type") == "bylaw_section":
            metadata["section_id"] = chunk.get("section_id", "N/A")
            metadata["title"] = chunk.get("title", "N/A")

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk["content"]],
            metadatas=[metadata],
        )
        

print("\n===================================")
print("Vectorization complete.")
print(f"Stored in: {VECTORDB_DIR}")
print("=====================================\n")