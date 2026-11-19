"""Sections 4-6: PDF extraction (with page numbers), cleaning, PII removal, chunking."""
import re
import pandas as pd
from pypdf import PdfReader
from .config import PATHS, CONFIG

# ---------------- PII patterns ----------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone numbers: +91 98765 43210, 020-12345678, 9876543210 (8+ digits are masked)
PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s\-]?)?(?:\d{5}[\s\-]\d{5}|\(?0\d{2,4}\)?[\s\-]?\d{6,8}|\d{10})(?!\d)")
# "Roll No: 21CS045", "Seat Number 123456", "PRN: 72012345K"
ROLL_RE = re.compile(r"(?i)\b(roll|seat|prn|enrol(?:l)?ment|reg(?:istration)?)\s*(?:no|number|num|id)?\.?\s*[:\-]?\s*[A-Z0-9/\-]{4,}")


def anonymize(text: str) -> str:
    """Replace emails, phone numbers and roll/seat/PRN numbers with placeholders.
    NOTE: person names are NOT removed automatically (needs NER). Check your PDFs manually."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = ROLL_RE.sub(lambda m: m.group(1) + " [ID]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return text


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)          # join words hyphenated across lines
    text = re.sub(r"\s*\n\s*", " ", text)           # newlines -> spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_pages(pdf_path):
    """Return list of dicts {doc_name, page_no, text}. page_no is the physical PDF page (1-based)."""
    rows = []
    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                print(f"[SKIP] {pdf_path.name}: password-protected")
                return rows
        for i, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                raw = ""
            rows.append({"doc_name": pdf_path.name, "page_no": i,
                         "text": anonymize(clean_text(raw))})
    except Exception as e:
        print(f"[FAIL] {pdf_path.name}: {e}")
    return rows


def chunk_words(text, size, overlap):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(save=True, verbose=True):
    """PDFs in data/raw_pdfs -> chunks.csv. Chunks never cross page boundaries,
    so every citation (document + page) is exact."""
    pdfs = sorted(PATHS["raw"].glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {PATHS['raw']}. Put your public/permitted PDFs there.")
    pages = []
    for p in pdfs:
        rows = extract_pages(p)
        empty = sum(1 for r in rows if len(r["text"].split()) < 5)
        if verbose:
            print(f"{p.name}: {len(rows)} pages, {empty} pages with (almost) no text")
        if rows and empty / len(rows) > 0.5:
            print(f"  [WARN] {p.name} looks scanned (image-only). Text extraction needs OCR.")
        pages.extend(rows)
    pages_df = pd.DataFrame(pages)

    chunk_rows = []
    for r in pages:
        for c in chunk_words(r["text"], CONFIG["chunk_words"], CONFIG["chunk_overlap"]):
            if len(c.split()) >= CONFIG["min_chunk_words"]:
                chunk_rows.append({"doc_name": r["doc_name"], "page_no": r["page_no"], "text": c})
    chunks_df = pd.DataFrame(chunk_rows)
    if chunks_df.empty:
        raise ValueError("No usable text extracted. PDFs may be scanned images.")
    chunks_df.insert(0, "chunk_id", range(len(chunks_df)))
    if save:
        pages_df.to_csv(PATHS["processed"] / "pages.csv", index=False)
        chunks_df.to_csv(PATHS["chunks"], index=False)
    if verbose:
        print(f"Total: {len(pdfs)} PDFs, {len(pages_df)} pages, {len(chunks_df)} chunks")
    return chunks_df
