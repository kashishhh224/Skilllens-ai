import io
from typing import Dict, List

try:
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        # Model is not installed; name extraction will gracefully degrade.
        _NLP = None
except ImportError:
    _NLP = None

# Optional PDF reader: prefer PyPDF2, fall back to pypdf if available.
try:
    from PyPDF2 import PdfReader as _PdfReader
except ImportError:  # pragma: no cover - environment dependent
    try:
        from pypdf import PdfReader as _PdfReader  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - environment dependent
        _PdfReader = None  # type: ignore[assignment]

# Optional DOCX reader (python-docx).
try:
    from docx import Document as _DocxDocument
except ImportError:  # pragma: no cover - environment dependent
    _DocxDocument = None


SECTION_KEYWORDS = {
    "skills": [
        "skills",
        "technical skills",
        "skills & abilities",
        "key skills",
    ],
    "education": [
        "education",
        "academic background",
        "academics",
    ],
    "projects": [
        "projects",
        "personal projects",
        "academic projects",
        "technical projects",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
    ],
}


def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """
    Extract text from a PDF file-like object using PyPDF2.
    """
    if _PdfReader is None:
        raise RuntimeError(
            "PDF extraction requires either 'PyPDF2' or 'pypdf' to be installed. "
            "Install it with 'pip install PyPDF2' or 'pip install pypdf'."
        )

    reader = _PdfReader(file_stream)
    texts: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            page_text = ""
        texts.append(page_text)
    return "\n".join(texts)


def extract_text_from_docx(file_stream: io.BytesIO) -> str:
    """
    Extract text from a DOCX file-like object using python-docx.
    """
    if _DocxDocument is None:
        raise RuntimeError(
            "DOCX extraction requires the 'python-docx' package. "
            "Install it with 'pip install python-docx'."
        )

    document = _DocxDocument(file_stream)
    paragraphs = [p.text for p in document.paragraphs if p.text]
    return "\n".join(paragraphs)


def extract_text_from_resume(file_bytes: bytes, extension: str) -> str:
    """
    Dispatch text extraction based on file extension.
    """
    stream = io.BytesIO(file_bytes)
    ext = extension.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(stream)
    if ext == ".docx":
        return extract_text_from_docx(stream)
    raise ValueError(f"Unsupported file extension: {extension}")


def _extract_name_with_spacy(text: str) -> str:
    if not _NLP:
        return ""
    doc = _NLP(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return ""


def _fallback_name(lines: List[str]) -> str:
    for line in lines:
        if line.strip():
            return line.strip()
    return ""


def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Parse a raw resume text blob into coarse sections using simple
    heuristic heading detection. Returns a dictionary with keys:
    name, skills, education, projects, experience, raw_text.
    """
    lines = [line.strip() for line in text.splitlines()]

    name = _extract_name_with_spacy(text)
    if not name:
        name = _fallback_name(lines)

    sections: Dict[str, str] = {
        "name": name,
        "skills": "",
        "education": "",
        "projects": "",
        "experience": "",
        "raw_text": text,
    }

    current_section_key = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower().rstrip(":")
        matched_key = None

        for key, keywords in SECTION_KEYWORDS.items():
            if any(lower.startswith(keyword) for keyword in keywords):
                matched_key = key
                break

        if matched_key:
            current_section_key = matched_key
            continue

        if current_section_key and line:
            sections[current_section_key] += line + "\n"

    return sections

