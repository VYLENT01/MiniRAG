"""
PDF Loader Implementation.

Extracts text from PDF files using pypdf.
Educational note: PDFs do not store text in logical reading order,
so extracted text might require aggressive cleaning later.
"""

from pathlib import Path

from minirag.exceptions import LoaderError
from minirag.loaders.base import BaseLoader

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError(
        "pypdf is required to load PDF files. "
        "Please install it via: pip install pypdf"
    )


class PDFLoader(BaseLoader):
    """Loads text content from PDF files."""

    def load(self, file_path: Path) -> str:
        if not file_path.exists():
            raise LoaderError(f"File not found: {file_path}")

        try:
            reader = PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            if not text_parts:
                raise LoaderError(f"No text could be extracted from PDF: {file_path}")
                
            return "\n".join(text_parts)
        except Exception as e:
            raise LoaderError(f"Failed to parse PDF {file_path}: {e}") from e