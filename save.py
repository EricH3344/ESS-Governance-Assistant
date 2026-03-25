from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import spacy

# ---------------------------------------------------------
# Load spaCy
# ---------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------
# spaCy enrichment
# ---------------------------------------------------------
def enrich(content: str) -> Dict[str, Any]:
    doc = nlp(content)
    return {
        "content": content,
        "sentences": [s.text.strip() for s in doc.sents if s.text.strip()],
        "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
    }


# ---------------------------------------------------------
# Detect document type
# ---------------------------------------------------------
def detect_doc_type(text: str) -> str:
    t = text[0:100].lower()
    if "meeting minutes" in t:
        return "minutes"
    if "policy" in t:
        return "policy"
    if "by-laws" in t:
        return "bylaws"
    return "unknown"


# ---------------------------------------------------------
# Section header detection
# ---------------------------------------------------------
def detect_section_header(line: str) -> Optional[str]:
    lower = line.lower()

    if "land ack" in lower:
        return "land_ack"
    if "officers present" in lower or "officers excus" in lower:
        return "attendance"
    if "updates" in lower and "mise" in lower:
        return "minutes_update"
    if "discussion items" in lower or "éléments de discussion" in lower:
        return "discussion"
    if "motion" in lower:
        return "motion"
    if "varia" in lower:
        return "varia"
    if "reminders" in lower:
        return "reminders"

    return None


# ---------------------------------------------------------
# Extract roles from attendance (Option B)
# ---------------------------------------------------------
def extract_role_from_attendance_line(line: str) -> Optional[tuple[str, str]]:
    """
    Extract (role, person) from attendance lines like:
    * VP External - Ashna (she/her)
    """
    stripped = line.strip()
    if not stripped.startswith("*"):
        return None

    if " - " not in stripped:
        return None

    # Remove leading "*"
    body = stripped[1:].strip()

    role, rest = body.split(" - ", 1)
    role = role.strip()

    # Person name is before any "(" or trailing notes
    person = rest.split("(")[0].strip()
    person = person.replace(" - excused", "").strip()

    return role, person



# ---------------------------------------------------------
# Role header detection using roles_in_meeting (fuzzy match)
# ---------------------------------------------------------
def normalize_header_candidate(line: str) -> str:
    s = line.strip()
    # strip bullet
    if s.startswith("*") or s.startswith("-"):
        s = s[1:].strip()
    # strip trailing colon
    if s.endswith(":"):
        s = s[:-1].strip()
    return s


def match_role_header(line: str, roles_in_meeting: dict[str, str]) -> Optional[str]:
    """
    Fuzzy match a header line to a role from attendance.
    """
    candidate = normalize_header_candidate(line)
    if not candidate:
        return None

    # 1) Exact match
    for role in roles_in_meeting:
        if candidate == role:
            return role

    # 2) Prefix / containment match (your original logic)
    for role in roles_in_meeting:
        if candidate.startswith(role) or role.startswith(candidate):
            return role

    # 3) NEW: substring fuzzy match
    #    "President" in "Co-President"
    #    "Finance" in "VP Finance & Administration"
    #    "Sponsorship" in "Manager of Sponsorship"
    cand_lower = candidate.lower()
    for role in roles_in_meeting:
        role_lower = role.lower()
        if cand_lower in role_lower or role_lower in cand_lower:
            return role

    return None


# ---------------------------------------------------------
# Minutes parsing: sections + roles (Option C)
# ---------------------------------------------------------
def extract_minutes_chunks(text: str, source: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    roles_in_meeting: dict[str, str] = {}

    current_section_type: str = "doc_start"
    current_role: Optional[str] = None
    current_lines: List[str] = []
    in_attendance_section = False

    def flush():
        nonlocal current_lines, current_role, current_section_type
        if not current_lines:
            return
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        enriched = enrich(content)
        chunks.append({
            "type": current_section_type,
            "document_type": "minutes",
            "source_file": source,
            "role": current_role if current_role is not None else "N/A",
            "person": roles_in_meeting.get(current_role, "N/A"),
            **enriched,
        })
        current_lines = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # 1) Section header?
        sec_type = detect_section_header(line)
        if sec_type is not None:
            flush()
            current_section_type = sec_type
            current_role = None
            in_attendance_section = (sec_type == "attendance")
            continue  # do not include section header in content

        # 2) Attendance section → extract roles, but keep as one chunk
        if in_attendance_section:
            extracted = extract_role_from_attendance_line(line)
            if extracted:
                role, person = extracted
                roles_in_meeting[role] = person
            current_lines.append(line.strip())
            continue

        # 4) Role header? (only outside attendance)
        matched_role = match_role_header(line, roles_in_meeting)
        if matched_role is not None and current_section_type != "attendance":
            flush()
            current_role = matched_role
            continue  # do not include role header itself in content

        # 5) Otherwise → content
        current_lines.append(line.strip())

    # Flush last chunk
    flush()

    return chunks

# ---------------------------------------------------------
# Policy / Bylaws section extraction
# ---------------------------------------------------------

SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$", re.MULTILINE)

def extract_policy_sections(text: str, source: str, doc_type: str) -> List[Dict[str, Any]]:
    chunks = []
    matches = list(SECTION_RE.finditer(text))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_id = m.group(1)
        title = m.group(2)
        block = text[start:end].strip()

        enriched = enrich(block)

        chunks.append({
            "type": "policy_section",
            "document_type": doc_type,
            "source_file": source,
            "section_id": section_id,
            "title": title,
            **enriched,
        })

    return chunks

# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def preprocess_document(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    doc_type = detect_doc_type(text)

    if doc_type == "minutes":
        return extract_minutes_chunks(text, path.name)

    if doc_type in ("policy", "bylaws"):
        return extract_policy_sections(text, path.name, doc_type)

    enriched = enrich(text.strip())
    return [{
        "type": "raw_document",
        "document_type": doc_type,
        "source_file": path.name,
        **enriched,
    }]
