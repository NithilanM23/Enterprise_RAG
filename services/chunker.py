"""
services/chunker.py
--------------------
Text chunking with section heading enrichment.

Approach:
  1. Split the text on [HEADING: ...] markers to get named sections
  2. Chunk each section independently using RecursiveCharacterTextSplitter
  3. Prepend [Section: HeadingName] to every chunk from that section

This is simpler and faster than position-mapping approaches — no scanning,
no offset arithmetic, no risk of hanging on large documents.

Result per chunk:
  "[Section: Products]\n-CMS software -CRM software -ERP software..."

The LLM sees the section label and can confidently answer topic-specific
questions even when the chunk content is a raw list or fragment.
"""

import logging
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

HEADING_PATTERN = re.compile(r'\[HEADING:\s*(.+?)\]')


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> list:
    """
    Split text into sections using [HEADING: ...] markers as dividers.

    Returns:
        List of (heading, content) tuples.
        heading is "" for content before the first heading.
        Content has marker text removed.
    """
    sections   = []
    last_end   = 0
    last_head  = ""

    for match in HEADING_PATTERN.finditer(text):
        # Collect text between previous heading and this one
        content = text[last_end:match.start()].strip()
        if content:
            sections.append((last_head, content))

        last_head = match.group(1).strip()
        last_end  = match.end()

    # Remaining text after the last heading
    tail = text[last_end:].strip()
    if tail:
        sections.append((last_head, tail))

    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list:
    """
    Split raw text into overlapping chunks with section heading enrichment.

    Steps:
      1. Split on [HEADING: ...] markers → named sections
      2. Chunk each section with RecursiveCharacterTextSplitter
      3. Prefix each chunk with [Section: heading] if a heading exists

    Args:
        text: Raw document text with embedded [HEADING: ...] markers
              (produced by loader.py).

    Returns:
        List of dicts:
            {
                "chunk_number": int,
                "chunk_text":   str,   # enriched with section heading
                "embedding":    None
            }

    Raises:
        ValueError: If text is empty or produces no non-empty chunks.
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split into sections
    sections = _split_into_sections(text)

    if not sections:
        # No heading markers found — chunk the whole text as one section
        sections = [("", text)]

    all_chunks    = []
    chunk_counter = 0

    for heading, content in sections:
        if not content.strip():
            continue

        raw_chunks = splitter.split_text(content)
        raw_chunks = [c for c in raw_chunks if c.strip()]

        for raw in raw_chunks:
            # Prefix with section heading if one exists
            if heading:
                enriched = f"[Section: {heading}]\n{raw}"
            else:
                enriched = raw

            all_chunks.append({
                "chunk_number": chunk_counter,
                "chunk_text":   enriched,
                "embedding":    None,
            })
            chunk_counter += 1

    if not all_chunks:
        raise ValueError(
            "Text could not be split into any non-empty chunks. "
            "Check that the document contains readable content."
        )

    heading_count = sum(1 for h, _ in sections if h)
    logger.info(
        "Chunked into %d chunks across %d sections "
        "(chunk_size=%d, overlap=%d).",
        len(all_chunks), heading_count,
        CHUNK_SIZE, CHUNK_OVERLAP,
    )

    return all_chunks


def get_chunk_config() -> dict:
    return {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}