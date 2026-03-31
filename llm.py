import requests
from retrieval import retrieve, get_meeting_index, resolve_meeting_date, detect_document_type

# ---------------------------------------------------------
# Build the context from retrieved chunks
#
# Function: format the retrieved chunks into a single 
# context string for the LLM, including metadata for each chunk
# 
# Returns: a formatted context string that combines the content 
# and metadata of the retrieved chunks
# ---------------------------------------------------------
def build_context(chunks: list[dict]) -> str:
    parts = []
    # Create a labeled section with metadata for each chunk
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
            f"TYPE: {stype} | PERSON: {person} | ROLE: {role} ---\n"
            f"CONTENT:\n{c['content']}\n"
        )
        parts.append(label)
    # Join all parts with clear separators
    return "\n\n====================\n\n".join(parts)


# System prompt for the LLM
SYSTEM_PROMPT = """\
You are a governance assistant for the University of Ottawa Engineering Students' Society (ESS). 
Your task is to answer the user's question using ONLY the provided CONTEXT.

Rules:
1. ALWAYS assume the provided CONTEXT contains the most relevant and recent information \
   requested by the user. If the user asks for the "last" or "most recent" meeting, \
   use the most recent date found in the CONTEXT.
2. When the user asks about governance rules prefer policy/bylaw content over meeting minutes.
3. If the user asks what happened in a meeting, summarize in point form:
   - key discussions
   - decisions made
   - motions passed
   - action items
5. ALWAYS prioritize substantive content over headers or summaries.
6. NO INTRODUCTIONS. Output only the answer.
7. If multiple points exist, present them as bullet points.
8. NEVER state that information is "not explicitly mentioned" or "not available" if there is relevant data in the CONTEXT.
9. NEVER assume acronyms or abbreviations unless they are explicitly defined in the CONTEXT.
10. NO INTRODUCTIONS or FILLER. Get straight to the details.
11. Names and accents are strict: "Zoe" and "Zoé" are different people.
12. Map abbreviations: "VP COMMS" -> "VP Communications", etc.
13. If the question does not include a specific document type, look for the best match across all document types. If the question is about a specific document type, only use that type for answering.
14. ALWAYS respond in English, unless the prompt is in French, then respond in French. Do not switch languages in the same response.
15. If the user prompt contains a date, convert it to ISO format (YYYY-MM-DD), assuming the current year, and use it to find relevant chunks.
"""

# ---------------------------------------------------------
# Build the final prompt for the LLM
#
# Function: combine the user query with the retrieved 
# context to create a final prompt for the LLM
# 
# Returns: a string that includes instructions, 
# the context, and the user query
# ---------------------------------------------------------
def build_prompt(query: str, context: str) -> str:
    return (
        f"Use the following CONTEXT to answer the QUERY.\n"
        f"Do not invent facts or assume details not present in CONTEXT.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUERY: {query}"
    )

# ---------------------------------------------------------
# Merge the retrieved chunks
#
# Function: combine and deduplicate the retrieved chunks from 
# both the hyde and original query retrievals, and sort 
# them by relevance score
# 
# Returns: a list of chunks sorted by relevance score
# ---------------------------------------------------------
def merge_chunks(chunks_a: list[dict], chunks_b: list[dict], k: int) -> list[dict]:
    seen = {}
    for c in chunks_a + chunks_b:
        cid = c["id"]
        if cid not in seen or c["score"] < seen[cid]["score"]:
            seen[cid] = c
    return sorted(seen.values(), key=lambda x: x["score"])[:k]

# ---------------------------------------------------------
# Choose the best chunks
#
# Function: based on the detected document type and meeting date, 
# choose the most relevant chunks for answering the query, with 
# a fallback to broader retrieval if needed
# 
# Returns: a tuple of (list of relevant chunks, detected document type)
# ---------------------------------------------------------
def choose_best_chunks(query: str, hyde: str | None, detected_doc_type: str | None, meeting_date: str | None, detected_doc_subtype: str | None, k: int, role: str | None = None) -> tuple[list[dict], str | None]:
    # If the query appears to be about policy or bylaws, restrict retrieval to that document type
    if detected_doc_type in ("policy", "bylaws"):
        return (
            merge_chunks(
                retrieve(hyde or query, k=k, document_type=detected_doc_type, role=role),
                retrieve(query, k=k, document_type=detected_doc_type, role=role),
                k,
            ),
            detected_doc_type,
        )

    # If the query is about a specific meeting or meeting subtype, restrict retrieval to minutes
    if meeting_date or detected_doc_subtype:
        return (
            merge_chunks(
                retrieve(hyde or query, k=k, document_type="minutes", document_subtype=detected_doc_subtype, meeting_date=meeting_date, role=role),
                retrieve(query, k=k, document_type="minutes", document_subtype=detected_doc_subtype, meeting_date=meeting_date, role=role),
                k,
            ),
            "minutes",
        )

    # Otherwise, search across all document types and choose the best result
    candidates = {}
    for doc_type in ["policy", "bylaws", "minutes"]:
        candidates[doc_type] = merge_chunks(
            retrieve(hyde or query, k=k, document_type=doc_type),
            retrieve(query, k=k, document_type=doc_type),
            k,
        )

    # ---------------------------------------------------------
    # Quality function
    #
    # Function: define a quality metric for the retrieved chunks 
    # based on the best score and number of chunks
    # 
    # Returns: a string that includes the content and 
    # metadata for better retrieval relevance
    # ---------------------------------------------------------
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
# Call the LLM
#
# Function: send the final prompt to the LLM and return the response
# 
# Returns: the LLM's answer to the user's query
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

# ---------------------------------------------------------
# Hypothetical Answer Generation (HyDE)
#
# Function: generate a hypothetical answer to the user's 
# query to improve retrieval
# 
# Returns: a short, plausible answer to the prompt based on
# the document type 
# ---------------------------------------------------------
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
        "temperature": 0.0,
    }
    response = requests.post("http://localhost:11434/v1/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------
# Answer the question
#
# Function: main function to answer the user's question by 
# detecting the document type, retrieving relevant chunks, 
# and calling the LLM with the appropriate context
# 
# Returns: the final answer to the user's question
# ---------------------------------------------------------
def answer_question(query: str, k: int = 10, model: str = "llama3.1:8b", document_type: str | None = None, role: str | None = None) -> str:
    detected_doc_type = detect_document_type(query)
    meeting_date, detected_doc_subtype = resolve_meeting_date(query)

    if not detected_doc_type and document_type:
        detected_doc_type = document_type

    # Set a higher k for meeting queries since they are more sparse and to give the model more to work with
    effective_k = 50 if meeting_date else k

    # Only use HyDE if there is no clear document type or meeting date
    hyde = None
    if not detected_doc_type and not meeting_date and not detected_doc_subtype:
        hyde = hypothetical_answer(query, model, doc_type=None)

    final_chunks, selected_doc_type = choose_best_chunks(
        query,
        hyde,
        detected_doc_type,
        meeting_date,
        detected_doc_subtype,
        effective_k,
        role=role,
    )

    # If the query is about policies/bylaws, restrict to those chunks only
    if selected_doc_type == "policy" or selected_doc_type == "bylaws":
        final_chunks = final_chunks[:effective_k]

    if not final_chunks:
        if detected_doc_type in ("policy", "bylaws"):
            return "I couldn't find relevant policy/bylaw sections for that question. Could you provide more detail or specify which governance area you mean?"

        final_chunks = merge_chunks(
            retrieve(hyde, k=effective_k),
            retrieve(query, k=effective_k),
            effective_k,
        )

    if not final_chunks:
        return "No relevant documents found. Could you provide more detail?"

    # Score threshold check
    best_score = final_chunks[0]["score"]
    if best_score > 0.44:
        return "I'm not confident I found the right information. Could you provide more detail?"

    # Generate the context and call the LLM
    context = build_context(final_chunks)
    prompt = build_prompt(query, context)
    return call_llm(prompt, model=model)