from pathlib import Path
import json
from preprocessor import preprocess_document

# ---------------------------------------------------------
# 1. Base directory = folder where THIS script is located
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------
# 2. Documents folder (where you put your test files)
# ---------------------------------------------------------
DOCS_DIR = BASE_DIR / "Documents"
DOCS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 3. Pick ONE file to test
# ---------------------------------------------------------
TEST_FILE = DOCS_DIR / "sample_minutes.txt"

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

print("\n==============================================")
print(f"📄 Processing file: {TEST_FILE.name}")
print("==============================================\n")

# ---------------------------------------------------------
# 4. Run preprocessing
# ---------------------------------------------------------
chunks = preprocess_document(TEST_FILE)

# ---------------------------------------------------------
# 5. Print all chunks to terminal
# ---------------------------------------------------------
for i, chunk in enumerate(chunks, start=1):
    print(f"---------------- Chunk {i} ----------------")
    print(f"Type: {chunk.get('type')}")
    print(f"Section/Role: {chunk.get('section_id') or chunk.get('role') or 'N/A'}")
    print(f"Person: {chunk.get('person')}")
    print("\nContent:")
    print(chunk.get("content"))
    print("--------------------------------------------\n")

print(f"Total chunks generated: {len(chunks)}\n")

# ---------------------------------------------------------
# 6. Save output to Processed/
# ---------------------------------------------------------
OUTPUT_DIR = BASE_DIR / "Processed"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"{TEST_FILE.stem}_chunks.json"

OUTPUT_FILE.write_text(
    json.dumps(chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("==============================================")
print(f"💾 Output saved to: {OUTPUT_FILE}")
print("==============================================\n")
