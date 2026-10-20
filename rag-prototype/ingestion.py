"""Document loading and chunking for PDF and TXT files."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

from pypdf import PdfReader

CHUNK_SIZE = 1000      # target characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks
MAX_FILE_MB = 10


class IngestionError(Exception):
    """Raised with a user-friendly message when a file cannot be processed."""


@dataclass
class Chunk:
    text: str
    source: str              # file name
    page: Optional[int]      # 1-based page number; None for TXT files


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages(name: str, data: bytes) -> list[tuple[Optional[int], str]]:
    """Return a list of (page_number, text). TXT files return page_number=None."""
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise IngestionError(f"'{name}' is larger than {MAX_FILE_MB} MB.")

    lower = name.lower()
    if lower.endswith(".txt"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")
        text = _clean(text)
        return [(None, text)] if text else []

    if lower.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise IngestionError(f"'{name}' is password-protected.")
            pages = []
            for i, page in enumerate(reader.pages, start=1):
                text = _clean(page.extract_text() or "")
                if text:
                    pages.append((i, text))
            return pages
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Could not read PDF '{name}': {exc}") from exc

    raise IngestionError(f"'{name}' is not a supported file type (use PDF or TXT).")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # look for a natural break in the last 40% of the window
            floor = int(len(window) * 0.6)
            for sep in ("\n\n", "\n", ". ", " "):
                idx = window.rfind(sep, floor)
                if idx != -1:
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def ingest_file(name: str, data: bytes) -> list[Chunk]:
    """Load one file and return its chunks. Raises IngestionError on problems."""
    pages = load_pages(name, data)
    if not pages:
        raise IngestionError(
            f"No text found in '{name}'. It may be a scanned/image-only PDF "
            "(OCR is not supported in this prototype) or an empty file."
        )
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for piece in split_text(text):
            chunks.append(Chunk(text=piece, source=name, page=page_no))
    return chunks
