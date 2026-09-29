from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(
    text : str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) ->List[str]:
    """Split text into chunks with overlap"""
    text = text.strip()
    if not text:
        return []

    if overlap >=chunk_size:
        raise ValueError("overlap must be smaller than chunk size")

    chunks=[]
    start= 0
    text_len= len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end>=text_len:
            break
        start = end - overlap
    return chunks
