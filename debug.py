import os
import re
from collections import defaultdict
import chromadb

# 🔧 YOUR PROJECT ROOT (hardcoded for simplicity)
PROJECT_DIR = "/Users/erichagen/Library/CloudStorage/OneDrive-UniversityofOttawa/Year 4/CEG 4195/Project"

# Chroma DB directory (default from your pipeline)
CHROMA_DIR = "/Users/erichagen/Library/CloudStorage/OneDrive-UniversityofOttawa/Year 4/CEG 4195/Project/vectordb"

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "debug")


def load_collection():
    print("🔄 Loading ChromaDB...")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collections = client.list_collections()
    print("📦 Collections found:", [c.name for c in collections])

    # 👇 adjust name if needed
    collection_date = collections[0].name
    collection = client.get_collection(collection_date)

    print(f"✅ Using collection: {collection_date}")
    return collection


def export_by_person(collection):
    print("🚀 Running export...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("📁 Saving to:", OUTPUT_DIR)

    data = collection.get(include=["documents", "metadatas"])

    print("📊 Total docs:", len(data["documents"]))

    if len(data["documents"]) == 0:
        print("❌ No data found in collection!")
        return

    # 🔍 peek at metadata
    print("\n🔍 Sample metadata:")
    for i in range(min(3, len(data["metadatas"]))):
        print(data["metadatas"][i])

    grouped = defaultdict(list)

    for doc, meta in zip(data["documents"], data["metadatas"]):
        date = meta.get("meeting_date")

        if not date:
            continue

        if isinstance(date, str):
            date = [date]

        for i in date:
            i = re.sub(r'[-–:]+$', '', i).strip()

            if not i:
                continue

            grouped[i].append((doc, meta))

    print("\n👥 Dates found:", len(grouped))

    if len(grouped) == 0:
        print("❌ No dates found — check metadata keys ('meeting_date' vs 'dates')")
        return

    # ✍️ write files
    for date, entries in grouped.items():
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', date)
        filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

        with open(filepath, "w", encoding="utf-8") as f:
            for doc, meta in entries:
                f.write(f"Date: {meta.get('meeting_date')}\n")
                f.write(f"Source: {meta.get('source')}\n")
                f.write(doc)
                f.write("\n\n" + "=" * 60 + "\n\n")

    print(f"\n✅ Exported {len(grouped)} people")


if __name__ == "__main__":
    collection = load_collection()
    export_by_person(collection)