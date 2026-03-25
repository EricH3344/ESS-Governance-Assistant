from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import spacy
from dateutil import parser as dateparser

nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------
# Meeting date parsing
# ---------------------------------------------------------
def parse_meeting_date(text: str) -> tuple[str, str] | tuple[None, None]:
    """
    Find the 'Date:' line in the document header and parse it.
    Returns (iso_date, display_date) e.g. ("2026-02-23", "February 23, 2026"),
    or (None, None) if no date line is found.
 
    Handles formats like:
        Date: February 23rd 2026 7:00pm
        Date: Aug 28, 2025 7:00 PM
    """
    for line in text.splitlines()[:20]:         # date is always near the top
        if line.lower().startswith("date:"):
            date_str = line.split(":", 1)[1].strip()
            try:
                dt = dateparser.parse(date_str, fuzzy=True)
                return dt.strftime("%Y-%m-%d"), dt.strftime("%B %-d, %Y")
            except Exception:
                pass
 
    return None, None

def enrich(content: str) -> Dict[str, Any]:
    doc = nlp(content)
    return {
        "content": content,
        "sentences": [s.text.strip() for s in doc.sents if s.text.strip()],
        "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
    }

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
# FIXED SECTION HEADER DETECTION
# ---------------------------------------------------------
def detect_section_header(line: str) -> Optional[str]:
    lower = line.strip().lower()

    if "land ack" in lower:
        return "land_ack"
    if "officers present" in lower or "officers excus" in lower:
        return "attendance"
    if "discussion items" in lower:
        return "discussion"
    if "motions" in lower:
        return "motion"
    if "varia" in lower:
        return "varia"
    if "reminders" in lower:
        return "reminders"

    return None

# ---------------------------------------------------------
# Attendance parsing
# ---------------------------------------------------------
def extract_role_from_attendance_line(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    if not stripped.startswith("*"):
        return None
    if " - " not in stripped:
        return None

    body = stripped[1:].strip()
    role, rest = body.split(" - ", 1)
    role = role.strip()

    person = rest.split("(")[0].strip()
    person = re.sub(r"\s*-\s*excused\b", "", person, flags=re.IGNORECASE).strip()
    
    return role, person

# ---------------------------------------------------------
# Role header detection (MINIMAL FIX)
# ---------------------------------------------------------
def normalize_header_candidate(line: str) -> str:
    s = line.strip()
    if s.startswith("*") or s.startswith("-"):
        s = s[1:].strip()
    if s.endswith(":"):
        s = s[:-1].strip()
    # Strip bilingual suffix: "VP Academic/Académique" → "VP Academic"
    s = s.split("/")[0].strip()
    return s

def match_role_header(line: str, roles_in_meeting: dict[str, str]) -> Optional[str]:
    candidate = normalize_header_candidate(line)
    if not candidate:
        return None

    # Exact match
    for role in roles_in_meeting:
        if candidate.lower() == role.lower():
            return role

    # Prefix match
    for role in roles_in_meeting:
        if candidate.lower().startswith(role.lower()) or role.lower().startswith(candidate.lower()):
            return role

    # Unknown role → treat as valid role header
    if line.strip().endswith(":"):
        return candidate

    return None

# ---------------------------------------------------------
# Minutes parsing (MINIMAL FIXES)
# ---------------------------------------------------------
def extract_minutes_chunks(text: str, source: str) -> List[Dict[str, Any]]:
    chunks = []
    roles_in_meeting = {}

    meeting_date, meeting_date_display = parse_meeting_date(text)

    current_section_type = "doc_start"
    current_role = None
    current_lines = []
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

        people = roles_in_meeting.get(current_role, [])
        person = ", ".join(people) if people else "N/A"

        chunk: Dict[str, Any] = {
            "type": current_section_type,
            "document_type": "minutes",
            "source_file": source,
            "role": current_role if current_role else "N/A",
            "person": person,
            **enriched,
        }
        if meeting_date:
            chunk["meeting_date"] = meeting_date
            chunk["meeting_date_display"] = meeting_date_display

        chunks.append(chunk)
        current_lines = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # Section header?
        sec_type = detect_section_header(line)
        if sec_type is not None:
            flush()
            current_section_type = sec_type
            current_role = None
            in_attendance_section = (sec_type == "attendance")
            continue

        # Attendance — collect into roles_in_meeting, emit one chunk per person at the end
        if in_attendance_section:
            extracted = extract_role_from_attendance_line(line)
            if extracted:
                role, person = extracted
                roles_in_meeting.setdefault(role, []).append(person)
                # Also register the bare role name (without qualifier prefix like "Co-")
                # so that a minutes header "President:" matches the "Co-President" entry.
                bare = re.sub(r"^(co-|co\s+|interim\s)",
                              "", role, flags=re.IGNORECASE).strip()
                if bare and bare.lower() != role.lower():
                    roles_in_meeting.setdefault(bare, [])
                    for p in roles_in_meeting[role]:
                        if p not in roles_in_meeting[bare]:
                            roles_in_meeting[bare].append(p)
            continue

        # Role header detection (allowed everywhere except motion)
        if current_section_type != "motion":
            matched_role = match_role_header(line, roles_in_meeting)
            if matched_role is not None:
                flush()
                current_role = matched_role

                # If we're in discussion or varia, KEEP the section type
                if current_section_type not in ("discussion", "varia"):
                    current_section_type = "minutes_update"

                continue

        current_lines.append(line.strip())

    flush()

    # Emit one chunk per person in attendance so names are searchable.
    # roles_in_meeting maps role -> list[person] to handle co-holders (e.g. Co-President).
    for role, persons in roles_in_meeting.items():
        for person in persons:
            content = f"{person} holds the role of {role}."
            chunk: Dict[str, Any] = {
                "type": "attendance",
                "document_type": "minutes",
                "source_file": source,
                "role": role,
                "person": person,
                **enrich(content),
            }
            if meeting_date:
                chunk["meeting_date"] = meeting_date
                chunk["meeting_date_display"] = meeting_date_display
            chunks.append(chunk)

    return chunks

# ---------------------------------------------------------
# Policy parsing unchanged
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