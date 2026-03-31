import pytest

import retrieval
from preprocessor import (
    parse_meeting_date,
    detect_doc_type,
    detect_section_header,
    extract_role_from_attendance_line,
    match_role_header,
    normalize_header_candidate,
    extract_minutes_chunks,
    extract_policy_sections,
)
from vectorization import build_embedding_text
from retrieval import (
    build_filters,
    detect_document_type,
    resolve_meeting_date,
    get_meeting_index,
)
from llm import (
    build_context,
    build_prompt,
    merge_chunks,
)

# ==========================================
# 1. PREPROCESSOR TESTS
# ==========================================

def test_parse_meeting_date():
    """Test that the system can extract a date from meeting header text."""
    text = "ESS Meeting\nDate: Feb 15, 2026\n"
    iso_date, _ = parse_meeting_date(text)
    assert iso_date == "2026-02-15"

def test_detect_doc_type():
    """Test that the system can correctly identify document types based on content."""
    doc_type, subtype = detect_doc_type("This is a policy about data privacy.")
    assert doc_type == "policy"
    assert subtype == "policy"

def test_detect_section_header():
    """Test that bilingual headers are correctly mapped to internal categories."""
    header = detect_section_header("Attendance – présence")
    assert header == "attendance"

def test_extract_role_from_attendance():
    """Test that names and roles are properly extracted from bullet points."""
    result = extract_role_from_attendance_line("* President - Maya Benhamou")
    assert result == ("President", "Maya Benhamou")

def test_match_role_header():
    """Test that the system can match various header formats to known roles."""
    roles_in_meeting = {
        "officer": [],
        "attendees": [],
        "President": [],
    }
    assert match_role_header("Attendees", roles_in_meeting) == "attendees"
    assert match_role_header("President", roles_in_meeting) == "President"
    assert match_role_header("Présent.e.s:", roles_in_meeting) == "Présent.e.s"

def test_normalize_header_candidate():
    """Test that header candidates are normalized for matching."""
    assert normalize_header_candidate("  VP Academic/Académique  ") == "VP Academic"
    assert normalize_header_candidate("* Manager of IT - Adi") == "Manager of IT - Adi"

def test_extract_minutes_chunks():
    """Test that chunks are correctly extracted from minutes text."""
    text = (
        "Attendance\n"
        "* President - Eric Hagen\n"
        "* VP Finance - Cyrus Choi\n"
        "Discussion\n"
        "The team discussed a financial update in sufficient detail to form a valid chunk.\n"
    )
    chunks = extract_minutes_chunks(text, "meeting.txt", "officer")
    assert any(chunk["type"] == "attendance_summary" for chunk in chunks)
    assert any("Eric Hagen" in chunk.get("content", "") for chunk in chunks)
    assert any(chunk.get("role") == "N/A" for chunk in chunks)

def test_extract_policy_sections():
    """Test that policy sections are correctly extracted with titles and IDs."""
    text = "1. Data Privacy\nContent about data privacy."
    sections = extract_policy_sections(text, "policy-2026-EN.txt", "policy")
    assert len(sections) == 1
    assert sections[0]["section_id"] == "1."
    assert sections[0]["title"] == "1. Data Privacy"
    assert "Content about data privacy." in sections[0]["content"]

# ==========================================
# 2. VECTORIZATION TESTS
# ==========================================

def test_build_embedding_text():
    """
    Validates that build_embedding_text creates the structured [tag] format
    required for the BGE model and avoids redundant content repetition.
    """
    mock_chunk = {
        "document_type": "minutes",
        "type": "minutes_chunk",
        "role": "VP Philanthropic",
        "person": "Zoe",
        "content": "1.5k in donations for Carty House"
    }

    result = build_embedding_text(mock_chunk)

    # 3. Assertions to ensure metadata is structured correctly
    assert "[document_type: minutes]" in result
    assert "[type: minutes_chunk]" in result
    assert "[role: VP Philanthropic]" in result
    assert "[person: Zoe]" in result
    
    # 4. Assert that content appears exactly once (to prevent the repetition bug)
    assert result.count("1.5k in donations for Carty House") == 1
    
    # 5. Check formatting: verify it ends with the actual content prose
    assert result.endswith("1.5k in donations for Carty House")

# ==========================================
# 3. RETRIEVAL TESTS
# ==========================================

def test_build_filters():
    """Test that database filters are constructed correctly."""
    filters = build_filters(document_type="minutes", document_subtype="bod")
    # Pytest will check if the "$and" operator was added to combine the filters
    assert "$and" in filters
    assert {"document_type": "minutes"} in filters["$and"]

def test_detect_document_type():
    """Test that the system routes queries to the correct document type."""
    doc_type = detect_document_type("What do the bylaws say about quorum?")
    assert doc_type == "bylaws"

def test_resolve_meeting_date(monkeypatch):
    """Test that the system can identify meeting dates from queries."""
    monkeypatch.setattr(retrieval, "get_meeting_index", lambda doc_subtype=None: ["2026-02-15", "2026-01-05"])
    date, subtype = resolve_meeting_date("What was discussed at the last board meeting?")
    assert date == "2026-02-15"
    assert subtype == "bod"

def test_get_meeting_index(monkeypatch):
    """Test that the meeting index returns valid dates for a given subtype."""
    fake_results = {
        "metadatas": [
            {"document_type": "minutes", "document_subtype": "bod", "meeting_date": "2026-02-15"},
            {"document_type": "minutes", "document_subtype": "bod", "meeting_date": "2025-11-20"},
        ]
    }
    monkeypatch.setattr(retrieval.collection, "get", lambda include: fake_results)
    dates = get_meeting_index("bod")
    assert dates == ["2026-02-15", "2025-11-20"]

# ==========================================
# 4. LLM TESTS
# ==========================================

def test_build_context():
    """Test that the context builder formats retrieved chunks correctly."""
    chunks = [
        {
            "metadata": {
                "document_type": "policy",
                "document_subtype": "",
                "section_type": "",
                "person": "N/A",
                "role": "N/A",
                "meeting_date": "",
                "meeting_date_display": "",
            },
            "content": "This section covers data privacy.",
        },
        {
            "metadata": {
                "document_type": "minutes",
                "document_subtype": "bod",
                "section_type": "discussion",
                "person": "N/A",
                "role": "N/A",
                "meeting_date": "2026-02-15",
                "meeting_date_display": "",
            },
            "content": "At the last board meeting, we discussed the new data privacy policy.",
        },
    ]
    context = build_context(chunks)
    assert "This section covers data privacy." in context
    assert "At the last board meeting, we discussed the new data privacy policy." in context

def test_build_prompt():
    """Test that the final prompt includes both the user query and the context."""
    query = "Who is the VP Finance?"
    context = "Cyrus is the VP Finance."
    
    prompt = build_prompt(query, context)
    
    assert query in prompt
    assert context in prompt

def test_merge_chunks():
    """Test that multiple chunks are merged into a single coherent context."""
    chunks_a = [
        {"id": "1", "score": 0.5, "content": "Chunk 1 content."},
        {"id": "2", "score": 0.4, "content": "Chunk 2 content."},
    ]
    chunks_b = [
        {"id": "2", "score": 0.3, "content": "Chunk 2 content."},
        {"id": "3", "score": 0.6, "content": "Chunk 3 content."},
    ]
    merged = merge_chunks(chunks_a, chunks_b, k=3)
    assert any(chunk["content"] == "Chunk 1 content." for chunk in merged)
    assert any(chunk["content"] == "Chunk 2 content." for chunk in merged)
    assert any(chunk["content"] == "Chunk 3 content." for chunk in merged)