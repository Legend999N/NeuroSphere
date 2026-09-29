from pathlib import Path
import pymupdf
from docx import Document
#extract text from a supported file
def extract_text(file_path: Path) -> str :
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf" :
            return extract_pdf(file_path)
        if suffix == ".docx" :
            return extract_docx(file_path)
        
        return extract_plain_text(file_path)
    except Exception as exc:
        print(f"[ERROR]: Could not read {file_path}: {exc}")
        return ""

def extract_pdf(file_path: Path) -> str:
    text_parts = []
    with pymupdf.open(file_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text:
                text_parts.append(text)
    return "\n".join(text_parts).strip()

def extract_docx(file_path: Path) -> str:
    doc = Document(file_path)
    paragraphs =[
        paragraph.text.strip()
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    ]
    return "\n\n".join(paragraphs).strip()

def extract_plain_text(file_path: Path) -> str:
    return file_path.read_text(
        encoding="utf-8", errors="ignore"
    ).strip()

    

