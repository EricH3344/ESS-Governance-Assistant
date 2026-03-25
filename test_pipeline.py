import sys
import json
from pathlib import Path

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"

def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return condition

def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


BASE_DIR     = Path(__file__).resolve().parent
DOCS_DIR     = BASE_DIR / "Documents"
PROCESSED_DIR = BASE_DIR / "Processed"
VECTORDB_DIR  = BASE_DIR / "vectordb"

all_passed = True

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PREPROCESSING — all documents in Documents/
# ═══════════════════════════════════════════════════════════════════════════════
section("1. Preprocessing  (Documents/)")

from preprocessor import preprocess_document, detect_doc_type

if not DOCS_DIR.exists():
    print(f"  {FAIL}  Documents/ folder not found at {DOCS_DIR}")
    sys.exit(1)

txt_files = sorted(DOCS_DIR.glob("*.txt"))
all_passed &= check("Documents/ folder exists and has .txt files",
                    len(txt_files) > 0, f"{len(txt_files)} files found")

PROCESSED_DIR.mkdir(exist_ok=True)
all_chunks_by_file = {}

for path in txt_files:
    chunks = preprocess_document(path)
    all_chunks_by_file[path.name] = chunks

    doc_type = detect_doc_type(path.read_text(encoding="utf-8", errors="ignore"))
    has_chunks = len(chunks) > 0
    all_passed &= check(f"{path.name}  →  {len(chunks)} chunks  (type={doc_type})",
                        has_chunks)

    # Save JSON for vectorization step
    out = PROCESSED_DIR / (path.stem + ".json")
    out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    # Extra checks for minutes files
    if doc_type == "minutes":
        dates   = {c.get("meeting_date") for c in chunks if c.get("meeting_date")}
        roles   = {c.get("role") for c in chunks if c.get("role") not in (None, "N/A")}
        persons = {c.get("person") for c in chunks if c.get("person") not in (None, "N/A")}

        all_passed &= check(f"  ↳ meeting date parsed",
                            len(dates) > 0, f"dates found: {sorted(dates)}")
        all_passed &= check(f"  ↳ roles extracted",
                            len(roles) > 0, f"{len(roles)} unique roles")
        all_passed &= check(f"  ↳ persons extracted",
                            len(persons) > 0, f"e.g. {sorted(persons)[:4]}")

print(f"\n  {INFO}  Processed JSON saved to {PROCESSED_DIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. VECTORIZATION
# ═══════════════════════════════════════════════════════════════════════════════
section("2. Vectorization  (ChromaDB)")

import chromadb
from sentence_transformers import SentenceTransformer

VECTORDB_DIR.mkdir(exist_ok=True)
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

chroma  = chromadb.PersistentClient(path=str(VECTORDB_DIR))
collection = chroma.get_or_create_collection(
    name="governance", metadata={"hnsw:space": "cosine"}
)

total_upserted = 0
for fname, chunks in all_chunks_by_file.items():
    for i, chunk in enumerate(chunks):
        chunk_id = f"{fname}_{i}"
        text = (
            f"[document_type: {chunk.get('document_type','unknown')}]\n"
            f"[type: {chunk.get('type','unknown')}]\n"
            f"[role: {chunk.get('role','N/A')}]\n"
            f"[person: {chunk.get('person','N/A')}]\n"
            f"{chunk['content']}"
        )
        embedding = embed_model.encode(
            "Represent this sentence for searching relevant passages: " + text
        ).tolist()

        metadata = {
            "source":               fname,
            "document_type":        chunk.get("document_type", "unknown"),
            "type":                 chunk.get("type", "unknown"),
            "role":                 chunk.get("role", "N/A"),
            "person":               chunk.get("person", "N/A"),
            "chunk_index":          i,
            "meeting_date":         chunk.get("meeting_date", ""),
            "meeting_date_display": chunk.get("meeting_date_display", ""),
        }
        if chunk.get("type") == "policy_section":
            metadata["section_id"] = chunk.get("section_id", "N/A")
            metadata["title"]      = chunk.get("title", "N/A")

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk["content"]],
            metadatas=[metadata],
        )
        total_upserted += 1

db_count = collection.count()
all_passed &= check(f"ChromaDB contains {db_count} chunks (upserted {total_upserted})",
                    db_count >= total_upserted)

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 55}")
if all_passed:
    print(f"  {PASS}  All tests passed.")
else:
    print(f"  {FAIL}  Some tests failed — see details above.")
print(f"{'═' * 55}\n")

sys.exit(0 if all_passed else 1)