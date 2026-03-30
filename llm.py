import requests
from retrieval import retrieve, get_meeting_index, resolve_meeting_date

# ---------------------------------------------------------
# 1. Build context from retrieved chunks
# ---------------------------------------------------------
def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        meta = c["metadata"]
        date = meta.get("meeting_date", "")
        date_display = meta.get("meeting_date_display", "")
        dtype = meta.get("document_type", "DOC").upper()
        subtype = meta.get("document_subtype", "").upper()
        stype = meta.get("section_type", "").upper()
        person = meta.get("person", "N/A")
        role = meta.get("role", "N/A")
        label = (
            f"--- SOURCE: {dtype} {subtype} | DATE: {date or date_display} | "
            f"TYPE: {stype} | PERSON: {person} | ROLE: {role} ---"
        )
        parts.append(f"{label}\n{c['content']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------
# 2. Build the LLM prompt
# ---------------------------------------------------------
SYSTEM_PROMPT = """\
You are a governance assistant for the Engineering Students' Society (ESS). 
Your task is to answer the user's question using ONLY the provided CONTEXT.

Rules:
1. ALWAYS assume the provided CONTEXT contains the most relevant and recent information \
   requested by the user. If the user asks for the "last" or "most recent" meeting, \
   use the most recent date found in the CONTEXT.
2. If the user asks what happened in a meeting, summarize in point form:
   - key discussions
   - decisions made
   - motions passed
   - action items
3. DO NOT return meeting metadata (date/time) unless explicitly asked.
4. ALWAYS prioritize substantive content over headers or summaries.
5. NO INTRODUCTIONS. Output only the answer.
6. If multiple points exist, present them as bullet points.
7. NEVER state that information is "not explicitly mentioned" or "not available" if there is relevant data in the CONTEXT. 
8. Do not assume acronyms or abbreviations unless they are explicitly defined in the CONTEXT.
9. NO INTRODUCTIONS or FILLER. Get straight to the details.
10. Names and accents are strict: "Zoe" and "Zoé" are different people.
11. Map abbreviations: "VP COMMS" -> "VP Communications", etc.
"""


def build_prompt(query: str, context: str) -> str:
    return (
        f"Use the following CONTEXT to answer the QUERY.\n"
        f"If the information isn't there, say \"Not found in the documents. Please provide more information.\"\n"
        f"Do not invent facts or assume details not present in CONTEXT.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUERY: {query}"
    )


def merge_chunks(chunks_a: list[dict], chunks_b: list[dict], k: int) -> list[dict]:
    seen = {}
    for c in chunks_a + chunks_b:
        cid = c["id"]
        if cid not in seen or c["score"] < seen[cid]["score"]:
            seen[cid] = c
    return sorted(seen.values(), key=lambda x: x["score"])[:k]


def choose_best_chunks(query: str, hyde: str, meeting_date: str | None, detected_doc_subtype: str | None, k: int) -> tuple[list[dict], str | None]:
    if meeting_date or detected_doc_subtype:
        return (
            merge_chunks(
                retrieve(hyde, k=k, document_type="minutes", document_subtype=detected_doc_subtype, meeting_date=meeting_date),
                retrieve(query, k=k, document_type="minutes", document_subtype=detected_doc_subtype, meeting_date=meeting_date),
                k,
            ),
            "minutes"
        )

    candidates = {}
    for doc_type in ["policy", "bylaws", "minutes"]:
        candidates[doc_type] = merge_chunks(
            retrieve(hyde, k=k, document_type=doc_type),
            retrieve(query, k=k, document_type=doc_type),
            k,
        )

    def quality(chunks: list[dict]) -> tuple[float, int]:
        if not chunks:
            return float("inf"), 0
        return chunks[0]["score"], -len(chunks)

    best_type = min(candidates.items(), key=lambda item: quality(item[1]))[0]
    final_chunks = candidates[best_type]

    if not final_chunks:
        final_chunks = merge_chunks(
            retrieve(hyde, k=k),
            retrieve(query, k=k),
            k,
        )
        return final_chunks, None

    return final_chunks, best_type


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


def hypothetical_answer(query: str, model: str, doc_type: str = None) -> str:
    if doc_type and doc_type.lower() != "unknown":
        context_prefix = f" regarding the {doc_type.replace('_', ' ')}"
    else:
        context_prefix = ""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                f"{context_prefix} Write a short plausible answer to this question as if you were "
                "reading from a governance document or meeting minutes. "
                "Be brief and specific. Do not say you don't know.\n\n"
                f"Question: {query}"
            ),
        }],
        "temperature": 0.2,
    }
    response = requests.post("http://localhost:11434/v1/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------
# 4. Main RAG answer function
# ---------------------------------------------------------
def answer_question(
    query: str,
    k: int = 20,
    model: str = "llama3.1:8b",
) -> str:
    # 1. Resolve metadata from query
    meeting_date, detected_doc_subtype = resolve_meeting_date(query)

    # 2. Determine search depth based on whether a specific date was found
    effective_k = 50 if meeting_date else k

    # 3. Generate HyDE (Hypothetical Document Embeddings) answer for better retrieval
    hyde = hypothetical_answer(query, model, doc_type=detected_doc_subtype)

    # 4. Retrieve the best chunks from the appropriate document type
    final_chunks, selected_doc_type = choose_best_chunks(
        query,
        hyde,
        meeting_date,
        detected_doc_subtype,
        effective_k,
    )

    # 5. If a policy document was selected, use the top two policy chunks
    # so the model gets both the section heading and the actual content.
    if selected_doc_type == "policy":
        final_chunks = final_chunks[:2]

    # 6. If no chunks were found, retry without any strict type filtering
    if not final_chunks:
        final_chunks = merge_chunks(
            retrieve(hyde, k=effective_k),
            retrieve(query, k=effective_k),
            effective_k,
        )

    if not final_chunks:
        return "No relevant documents found. Could you provide more detail?"

    # 7. Check confidence threshold — reject only if we got a BAD match (high distance score)
    best_score = final_chunks[0]["score"]
    print(f"DEBUG: best_score: {best_score}")

    if best_score > 0.8:
        return "I'm not confident I found the right information. Could you provide more detail?"

    # 8. Generate final response
    context = build_context(final_chunks)
    prompt = build_prompt(query, context)

    print(final_chunks)

    return call_llm(prompt, model=model)