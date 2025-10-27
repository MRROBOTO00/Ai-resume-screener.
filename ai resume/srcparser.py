# src/parser.py
import io
import pdfplumber
import fitz  # pymupdf

def extract_text_from_pdf_fileobj(file_obj):
    """
    Accepts a file-like object (BytesIO) or a path string.
    Returns extracted text.
    """
    # If file_obj is a path string, open normally using fitz
    if isinstance(file_obj, str):
        return extract_text_from_pdf_path(file_obj)

    # Try pdfplumber first (handles some PDFs better)
    try:
        file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages).strip()
            if text:
                return text
    except Exception:
        pass

    # Fallback to PyMuPDF (fitz)
    try:
        file_obj.seek(0)
        data = file_obj.read()
        doc = fitz.open(stream=data, filetype="pdf")
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text).strip()
    except Exception:
        return ""

def extract_text_from_pdf_path(path):
    """Extract from file path using fitz then fallback to pdfplumber."""
    try:
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text).strip()
    except Exception:
        try:
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages).strip()
        except Exception:
            return ""
