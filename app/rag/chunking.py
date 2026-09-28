"""Markdown-aware recursive chunking.

Splits a policy document on its `##`/`###` section headings first (so a chunk never
straddles two sections), then recursively splits any section still over the target size,
with a character-based overlap standing in for a token-based overlap.
"""

import re
from dataclasses import dataclass

CHARS_PER_TOKEN = 4
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class Chunk:
    text: str
    section_heading: str
    chunk_index: int


def split_into_sections(markdown_text: str) -> list[Section]:
    matches = list(_HEADING_RE.finditer(markdown_text))
    if not matches:
        return [Section(heading="", text=markdown_text.strip())]

    sections: list[Section] = []
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        if body:
            sections.append(Section(heading=heading, text=body))
    return sections


def _split_recursive(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE_CHARS:
        return [text]

    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            break
    else:
        parts = list(text)
        sep = ""

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= CHUNK_SIZE_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)

    # apply character-based overlap between consecutive chunks
    overlapped: list[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prev_tail = chunks[i - 1][-CHUNK_OVERLAP_CHARS:]
            overlapped.append(prev_tail + c)
    return overlapped


def chunk_document(markdown_text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for section in split_into_sections(markdown_text):
        for piece in _split_recursive(section.text):
            chunks.append(Chunk(text=piece, section_heading=section.heading, chunk_index=idx))
            idx += 1
    return chunks
