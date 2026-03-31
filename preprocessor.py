from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import spacy
from dateutil import parser as dateparser

# Load Spacy model once at the module level for efficiency
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------
# Meeting date parsing
#
# Function: extract meeting date from the first 50 lines of a 
# minutes document using regex and date parsing
# 
# Returns: tuple of ISO date string and original date string
# ---------------------------------------------------------
def parse_meeting_date(text: str) -> tuple[str | None, str | None]:
    for line in text.splitlines()[:50]:
        if not line.strip():
            continue
        lower = line.lower()
        if "date:" in lower or "meeting date" in lower:
            _, date_str = line.split(":", 1) if ":" in line else (None, line)
            date_str = date_str.strip() if date_str else line.strip()
            # Remove time and trailing noise to improve date parsing accuracy
            date_str = re.split(r"\s+\d{1,2}:\d{2}(?:\s*[ap]m)?", date_str, 1)[0].strip()
            # Remove trailing commas, dashes, and whitespace
            date_str = re.sub(r"[,–—-]+\s*$", "", date_str).strip()
            try:
                dt = dateparser.parse(date_str, fuzzy=True, dayfirst=False)
                if dt:
                    return dt.strftime("%Y-%m-%d"), date_str
            except Exception:
                continue
    return None, None

# ---------------------------------------------------------
# Enrichment function
#
# Function: extract sentences and entities from a text
# 
# Returns: dictionary containing the original content, 
# sentences, and entities
# ---------------------------------------------------------
def enrich(content: str) -> Dict[str, Any]:
    doc = nlp(content)
    return {
        "content": content,
        "sentences": [s.text.strip() for s in doc.sents if s.text.strip()],
        "entities": [{"text": e.text, "label": e.label_} for e in doc.ents],
    }

# ---------------------------------------------------------
# Document type detection
#
# Function: detect the type of a document from the first 
# 400 characters based on its content and filename
# 
# Returns: tuple of document type and subtype
# ---------------------------------------------------------
def detect_doc_type(text: str, filename: str = "") -> tuple[str, str]:
    t = text[0:400].lower()
    fn = filename.lower()

    if any(x in fn for x in ["bylaws", "by-laws"]) or "bylaw" in t or "by-laws" in t:
        return "bylaws", "bylaws"
    if "policy" in fn or "policy" in t:
        return "policy", "policy"
    if any(x in fn for x in ["officer", "executive"]) or "officer meeting" in t or "executive meeting" in t:
        return "minutes", "officer"
    if any(x in fn for x in ["bod", "board"]) or "bod meeting" in t or "board of directors" in t or "board meeting" in t:
        return "minutes", "bod"
    if "meeting minutes" in t or "meeting agenda" in t or "meeting notes" in t:
        return "minutes", "unknown"
    return "unknown", "unknown"


# ---------------------------------------------------------
# Section header detection
#
# Function: detect section headers in a minutes document
# 
# Returns: standardized section type or None if not a header
# ---------------------------------------------------------
def detect_section_header(line: str) -> Optional[str]:
    line_strip = line.strip()
    if not line_strip:
        return None
        
    lower = line_strip.lower()
    
    # If the line is very long, it's unlikely to be a header
    if len(line_strip) > 60:
        return None

    if "land ack" in lower:
        return "land_ack"
    # Parses the line using regex to match headers
    if re.match(r"^(officers\s+present|officers\s+excused|officers\s+absent|officers\s+présent|officers\s+excusé|attendance|attendees|present|absent|excused)\b", lower):
        return "attendance"
    if "updates" in lower and "mises" in lower:
        return "updates"
    if "discussion" in lower:
        return "discussion"
    if "motions" in lower:
        return "motion"
    if "varia" in lower:
        return "varia"
    if "reminders" in lower:
        return "reminders"
        
    return None

# ---------------------------------------------------------
# Role extraction and attendance parser
#
# Function: extract role and person from an attendance line
# 
# Returns: tuple of role and person
# ---------------------------------------------------------
def extract_role_from_attendance_line(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    # All attendance lines typically start with a bullet point 
    if not stripped.startswith("*"):
        return None
    # All attendance lines typically contain a " - " separator between role and name
    if " - " not in stripped:
        return None

    # Remove the bullet point and split on the first " - "
    body = stripped[1:].strip()
    role, rest = body.split(" - ", 1)
    role = role.strip()

    # Clean up the person string
    person = rest
    
    # Drop inline comments starting with parentheses or brackets 
    person = re.split(r"[\(\[]", person, 1)[0]
    
    # Drop trailing comments separated by a dashes
    person = re.split(r"\s+[-–—]\s+", person, 1)[0]
    
    # Drop comma-separated comments
    person = person.split(",")[0]
    
    # Clean up any leftover whitespace
    person = person.strip()
    
    return role, person

# ---------------------------------------------------------
# Header normalization
#
# Function: normalize header candidates by stripping whitespace, 
# bullets, colons, and bilingual suffixes
# 
# Returns: normalized header candidate string
# ---------------------------------------------------------
def normalize_header_candidate(line: str) -> str:
    s = line.strip()
    # Remove leading bullets and whitespace
    if s.startswith("*") or s.startswith("-"):
        s = s[1:].strip()
    if s.endswith(":"):
        s = s[:-1].strip()
    # Strip bilingual suffix: "VP Academic/Académique" → "VP Academic"
    s = s.split("/")[0].strip()
    return s


# ---------------------------------------------------------
# Header matching
#
# Function: match a header candidate to known roles in the 
# meeting, allowing for exact, prefix, and title-like matches
# 
# Returns: the matched role or None if no match
# ---------------------------------------------------------
def match_role_header(line: str, roles_in_meeting: dict[str, str]) -> Optional[str]:
    candidate = normalize_header_candidate(line)
    if not candidate:
        return None

    # Role headers are typically short
    if len(candidate) > 60:
        return None

    # Exact match
    for role in roles_in_meeting:
        if candidate.lower() == role.lower():
            return role

    # Prefix match
    for role in roles_in_meeting:
        if candidate.lower().startswith(role.lower()) or role.lower().startswith(candidate.lower()):
            return role

    # Unknown role headers often end with a colon and have no spaces in the first 20 characters
    if line.strip().endswith(":") and " " not in candidate[:20]:
        return candidate

    return None

# ---------------------------------------------------------
# Minutes parsing
#
# Function: extract structured chunks from minutes text, 
# including section type, role, person, and meeting date metadata
# 
# Returns: a list of chunks with metadata such as section type, 
# role, person, and meeting date
# ---------------------------------------------------------
def extract_minutes_chunks(text: str, source: str, doc_subtype: str) -> List[Dict[str, Any]]:
    chunks = []
    roles_in_meeting = {}

    meeting_date, meeting_date_display = parse_meeting_date(text)

    current_section_type = "doc_start"
    current_role = None
    current_lines = []

    # ---------------------------------------------------------
    # Flush function
    #
    # Function: flush the current buffer of lines into chunks with appropriate
    # 
    # Returns: None
    # ---------------------------------------------------------
    def flush():
        nonlocal current_lines, current_role, current_section_type
        if not current_lines:
            return

        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        
        # Determine the person(s) associated with the current role
        people = roles_in_meeting.get(current_role, [])
        person = ", ".join(people) if people else "N/A"

        # Keep the chunks for discussion and varia sections more like paragraphs
        if current_section_type in ("discussion", "varia"):
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [content]
            segments = paragraphs
        # Use Spacy sentence segmentation into smaller chunks for other sections
        else:
            # Filter out short sentences that are unlikely to be meaningful on their own
            segments = [s for s in enrich(content)["sentences"] if len(s) >= 20]

        for segment in segments:
            if len(segment) < 20:
                continue
            
            # Create a chunk for each segment with the appropriate metadata
            chunk: Dict[str, Any] = {
                "section_type": current_section_type,
                "type": "minutes_chunk",
                "document_type": "minutes",
                "document_subtype": doc_subtype or "unknown",
                "source_file": source,
                "role": current_role if current_role else "N/A",
                "person": person,
                "content": segment,
            }

            if meeting_date:
                chunk["meeting_date"] = meeting_date
                chunk["meeting_date_display"] = meeting_date_display

            chunks.append(chunk)

        current_lines = []

    # Process the text line by line to detect sections, roles, and content
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        # Detect section header
        sec_type = detect_section_header(line)
        if sec_type is not None:
            flush()
            current_section_type = sec_type
            current_role = None
            continue

        # Extract roles from attendance lines
        if current_section_type == "attendance":
            extracted = extract_role_from_attendance_line(line)
            if extracted:
                role, person = extracted
                # Store the role and associated person(s)
                roles_in_meeting.setdefault(role, []).append(person)
                # Remove things like "Co-" or "Interim" for standardization using regex
                bare = re.sub(r"^(co-|co\s+|interim\s)", "", role, flags=re.IGNORECASE).strip()
                # If the cleaned role matches a known role, associate the person with that role as well
                if bare and bare.lower() != role.lower():
                    roles_in_meeting.setdefault(bare, [])
                    for p in roles_in_meeting[role]:
                        if p not in roles_in_meeting[bare]:
                            roles_in_meeting[bare].append(p)
            continue

        # Detect role headers within discussion or varia sections
        if current_section_type != "motion" and not raw.startswith("   "):
            matched_role = match_role_header(line, roles_in_meeting)
            if matched_role is not None:
                flush()
                current_role = matched_role

                # If we're in discussion or varia, keep the section type
                if current_section_type not in ("discussion", "varia"):
                    current_section_type = "minutes_update"

                continue

        current_lines.append(line.strip())

    flush()
    
    # For the roles in the attendance section, create a summary chunk for them
    if roles_in_meeting:
        lines = [f"{p} is the {role}."
                 for role, persons in roles_in_meeting.items()
                 for p in persons]
        content = "From the meeting attendance, the officers include:\n" + "\n".join(lines)
        chunk: Dict[str, Any] = {
            "section_type": "attendance",
            "type": "attendance_summary",
            "document_type": "minutes",
            "document_subtype": doc_subtype or "unknown",
            "source_file": source,
            "role": "N/A",
            "person": "N/A",
            "content": content,
            **enrich(content),
        }
        if meeting_date:
            chunk["meeting_date"] = meeting_date
            chunk["meeting_date_display"] = meeting_date_display
        chunks.append(chunk)

    return chunks



# Regular expressions to identify section headers in policies
SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z].+)$", re.MULTILINE)
MAJOR_SECTION_RE = re.compile(r"^\s*\d+\.\s+[A-Z].+$")

# ---------------------------------------------------------
# Stripping the table of contents from policy documents
#
# Function: remove the table of contents from policy text
# to improve section parsing and reduce cluttered chunks
# 
# Returns: the text with the table of contents removed
# ---------------------------------------------------------
def strip_policy_toc(text: str) -> str:
    toc_match = re.search(r"^Table of Contents\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not toc_match:
        return text

    rest = text[toc_match.end():]
    end_match = re.search(r"\n\s*\n", rest)
    if end_match:
        return rest[end_match.end():]
    return rest

# ---------------------------------------------------------
# Policy section parsing
#
# Function: extract structured chunks from policy text based 
# on section headers, including section ID and title metadata
# 
# Returns: a list of chunks with metadata such as 
# section ID and title
# ---------------------------------------------------------
def extract_policy_sections(text: str, source: str, doc_type: str) -> List[Dict[str, Any]]:
    text = strip_policy_toc(text)
    chunks = []
    # Find all section headers in the text using the regex
    matches = list(SECTION_RE.finditer(text)) 

    # If there are less than 10 matches, try splitting headers 
    # and treat the whole block as a single section.
    if len(matches) < 10:
        current_title = None
        current_lines: List[str] = []

        # Process the text line by line to detect section headers and content
        for line in text.splitlines():
            if MAJOR_SECTION_RE.match(line) and not line.startswith('   '):
                if current_title is not None:
                    content = "\n".join(current_lines).strip()
                    if content:
                        enriched = enrich(content)
                        chunks.append({
                            "document_type": doc_type,
                            "type": "policy_section",
                            "source_file": source,
                            "section_id": current_title.split(" ", 1)[0],
                            "title": current_title,
                            **enriched,
                        })
                current_title = line.strip()
                current_lines = []
                continue
            if current_title is not None:
                current_lines.append(line)

        # Flush the last section if it exists
        if current_title is not None:
            content = "\n".join(current_lines).strip()
            if content:
                enriched = enrich(content)
                chunks.append({
                    "document_type": doc_type,
                    "type": "policy_section",
                    "source_file": source,
                    "section_id": current_title.split(" ", 1)[0],
                    "title": current_title,
                    **enriched,
                })

        return chunks
    
    # Process each section based on the detected headers
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_id = m.group(1)
        title = m.group(2)
        block = text[start:end].strip()

        enriched = enrich(block)

        chunks.append({
            "document_type": doc_type,
            "type": "policy_section",
            "source_file": source,
            "section_id": section_id,
            "title": title,
            **enriched,
        })

    return chunks

# ---------------------------------------------------------
# Preprocessing pipeline
#
# Function: preprocess a document by detecting its type 
# and extracting
# 
# Returns: a list of chunks with appropriate metadata 
# based on the document type
# ---------------------------------------------------------
def preprocess_document(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    doc_type, doc_subtype = detect_doc_type(text, path.name)

    if doc_type == "minutes":
        chunks = extract_minutes_chunks(text, path.name, doc_subtype)
        return chunks

    if doc_type in ("policy", "bylaws"):
        return extract_policy_sections(text, path.name, doc_type)

    enriched = enrich(text.strip())
    return [{
        "document_type": doc_type,
        "document_subtype": doc_subtype,
        "source_file": path.name,
        **enriched,
    }]