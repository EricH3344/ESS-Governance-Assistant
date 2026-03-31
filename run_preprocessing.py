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

    if "minutes" in doc_type:
        dates   = {c.get("meeting_date") for c in chunks if c.get("meeting_date")}
        roles   = {c.get("role") for c in chunks if c.get("role") not in (None, "N/A")}
        persons = {c.get("person") for c in chunks if c.get("person") not in (None, "N/A")}

print(f"\n  {INFO}  Processed JSON saved to {PROCESSED_DIR}")

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