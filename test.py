# check_db.py - run this once to verify
import chromadb
from pathlib import Path

client = chromadb.PersistentClient(path="/Users/erichagen/Library/CloudStorage/OneDrive-UniversityofOttawa/Year 4/CEG 4195/Project/vectordb")
collection = client.get_collection("governance")

results = collection.get(include=["metadatas", "documents"])

# Print all distinct meeting dates
dates = {}
for meta in results["metadatas"]:
    iso = meta.get("meeting_date", "")
    display = meta.get("meeting_date_display", "")
    if iso:
        dates[iso] = display

print("\n=== Charity Commissioner chunks across all meetings ===")
for doc, meta in zip(results["documents"], results["metadatas"]):
    if meta.get("role") == "Charity Commissioner":
        print(f"  [{meta.get('meeting_date_display')}] is_empty={meta.get('is_empty_update')} | {doc[:60]}")