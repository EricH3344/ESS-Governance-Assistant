import requests
from retrieval import retrieve, get_meeting_index, resolve_meeting_date


# ---------------------------------------------------------
# 1. Build context from retrieved chunks
# ---------------------------------------------------------
def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(No relevant documents were found.)"

    parts = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        role = meta.get("role", "Unknown")
        person = meta.get("person", "Unknown")
        doc_type = meta.get("document_type", "Unknown")
        section = meta.get("type", "Unknown")
        source = meta.get("source", "Unknown")
        date_display = meta.get("meeting_date_display", "")
        date_phrase = f" at the {date_display} meeting" if date_display else ""

        attribution = (
            f'The following was said by {person} (Role: {role}){date_phrase} '
            f'in a {doc_type} document (section: {section}, source: {source}).'
        )

        parts.append(
            f"--- CHUNK {i+1} ---\n"
            f"{attribution}\n\n"
            f"{c['content']}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------
# 2. Build the LLM prompt
# ---------------------------------------------------------
SYSTEM_PROMPT = """\
You are a governance assistant for the Engineering Students' Society (ESS).

You will be given a CONTEXT section containing excerpts from meeting minutes, \
policy documents, and bylaws. Answer the user's question using information \
found in the CONTEXT.

Rules:
- Each chunk in the context begins with a sentence stating exactly who said it \
and what their role is. Use that attribution literally and do not modify names.
- Names are case and accent-sensitive. "Zoe" and "Zoé" are two \
completely different people. Never merge, equate, or parenthesize them.
- A person's name may include a suffix like "- Excused" or "- Term Over". \
Strip the suffix when referring to them (e.g. "Maya - Excused for Winter \
Semester" refer to them as "Maya").
- When asked for 'last' or 'most recent' information, identify the most recent \
chunk that contains substantive information.
- If the answer is genuinely not in the context, respond with the best \
guess based on the context, but preface it with "Based on the context,...". \
"""


def build_prompt(query: str, context: str, chunks: list[dict] | None = None) -> str:
    # --- Meeting date index (dynamic, from DB) ---
    meetings = get_meeting_index()
    if meetings:
        date_lines = "\n".join(f"  - {m['display']} ({m['iso']})" for m in meetings)
        date_block = f"KNOWN MEETING DATES (most recent first):\n{date_lines}\n\n"
    else:
        date_block = ""

    # --- Roster ---
    roster = ""
    if chunks:
        seen = set()
        lines = []
        for c in chunks:
            meta = c["metadata"]
            person = meta.get("person", "Unknown")
            role = meta.get("role", "Unknown")
            key = (person, role)
            if key not in seen:
                seen.add(key)
                lines.append(f"  - {person} → {role}")
        if lines:
            roster = "PEOPLE IN THIS CONTEXT:\n"
            roster += "\n".join(lines) + "\n\n"

    return (
        f"{date_block}"
        f"{roster}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )


# ---------------------------------------------------------
# 3. Call the local LLM
# ---------------------------------------------------------
def call_llm(prompt: str, model: str = "llama3") -> str:
    url = "http://localhost:11434/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def hypothetical_answer(query: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                "Write a short plausible answer to this question as if you were "
                "reading from meeting minutes or a governance document. "
                "Be brief and specific. Do not say you don't know.\n\n"
                f"Question: {query}"
            ),
        }],
        "temperature": 0.5,
    }
    response = requests.post("http://localhost:11434/v1/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------
# 4. Main RAG answer function
# ---------------------------------------------------------
def answer_question(
    query: str,
    k: int = 5,
    document_type: str = None,
    role: str = None,
    meeting_date: str = None,
    model: str = "llama3",
) -> str:
    # Resolve any temporal reference in the query BEFORE retrieval
    # e.g. "last meeting" → "2026-02-23", passed as a Chroma filter
    # If the caller already supplied a date, respect it; otherwise auto-resolve
    if meeting_date is None:
        meeting_date = resolve_meeting_date(query)

    if meeting_date is None:
        effective_k = len(get_meeting_index()) * 3
    else:
        effective_k = k

    # HyDE retrieval — hypothetical answer drives embedding search
    hyde = hypothetical_answer(query, model)
    hyde_chunks = retrieve(hyde, k=effective_k, document_type=document_type, role=role, meeting_date=meeting_date)

    # Direct retrieval — raw query as a second signal
    direct_chunks = retrieve(query, k=effective_k, document_type=document_type, role=role, meeting_date=meeting_date)

    # Merge and deduplicate by chunk id, keeping the best (lowest) cosine score
    seen = {}
    for c in hyde_chunks + direct_chunks:
        if c["id"] not in seen or c["score"] < seen[c["id"]]["score"]:
            seen[c["id"]] = c
    chunks = sorted(seen.values(), key=lambda x: x["score"])[:k]

    context = build_context(chunks)
    prompt = build_prompt(query, context, chunks=chunks)
    return call_llm(prompt, model=model)