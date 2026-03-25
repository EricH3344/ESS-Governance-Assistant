from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "Processed"
VECTORDB_DIR = BASE_DIR / "vectordb"

PROCESSED_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

chroma = chromadb.PersistentClient(path=str(VECTORDB_DIR))
collection = chroma.get_or_create_collection(
    name="governance",
    metadata={"hnsw:space": "cosine"}
)

def build_embedding_text(chunk: dict) -> str:
    doc_type = chunk.get("document_type", "unknown")
    chunk_type = chunk.get("type", "unknown")
    role = chunk.get("role", "N/A")
    person = chunk.get("person", "N/A")
    meeting_date_display = chunk.get("meeting_date_display", "")

    header_parts = [
        f"[document_type: {doc_type}]",
        f"[type: {chunk_type}]",
    ]

    # Minutes-specific context
    if doc_type == "minutes":
        header_parts.append(f"[role: {role}]")
        header_parts.append(f"[person: {person}]")
        if meeting_date_display:
            header_parts.append(f"[meeting_date: {meeting_date_display}]")
            
    # Policy/bylaws-specific context
    if chunk_type == "policy_section":
        section_id = chunk.get("section_id", "N/A")
        title = chunk.get("title", "N/A")
        header_parts.append(f"[section_id: {section_id}]")
        header_parts.append(f"[title: {title}]")

    header = "\n".join(header_parts)
    return f"{header}\n{chunk['content']}"

for json_file in PROCESSED_DIR.glob("*.json"):
    print(f"📄 Loading: {json_file.name}")
    chunks = json.loads(json_file.read_text(encoding="utf-8"))

    for i, chunk in enumerate(chunks):
        source = chunk.get("source_file", json_file.name)
        chunk_id = f"{source}_{i}"

        text_for_embedding = build_embedding_text(chunk)
        embedding = model.encode(text_for_embedding).tolist()

        metadata = {
            "source": source,
            "document_type": chunk.get("document_type", "unknown"),
            "type": chunk.get("type", "unknown"),
            "role": chunk.get("role", "N/A"),
            "person": chunk.get("person", "N/A"),
            "chunk_index": i,
            "meeting_date":         chunk.get("meeting_date", ""),
            "meeting_date_display": chunk.get("meeting_date_display", ""),
        }

        # Policy/bylaws extras
        if chunk.get("type") == "policy_section":
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