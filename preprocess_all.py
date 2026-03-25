"""
preprocess_all.py
-----------------
Step 1 of the pipeline.

Reads every .txt file from the Documents/ folder, runs it through
preprocess_document(), and writes one JSON file per document into
the Processed/ folder.

Run:
    python preprocess_all.py
"""

from pathlib import Path
import json
from preprocessor import preprocess_document

BASE_DIR      = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "Documents"
PROCESSED_DIR = BASE_DIR / "Processed"

PROCESSED_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {".txt"}   # add ".md" etc. here if needed

def main():
    files = [f for f in DOCUMENTS_DIR.iterdir()
             if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print(f"⚠️  No supported files found in {DOCUMENTS_DIR}")
        return

    print(f"Found {len(files)} file(s) in {DOCUMENTS_DIR}\n")

    for path in sorted(files):
        print(f"📄 Preprocessing: {path.name}")
        try:
            chunks = preprocess_document(path)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

        out_path = PROCESSED_DIR / (path.stem + ".json")
        out_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"   ✅ {len(chunks)} chunks → {out_path.name}")

    print(f"\nPreprocessing complete. JSON files are in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()