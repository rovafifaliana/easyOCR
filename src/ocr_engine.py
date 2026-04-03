"""
ocr_engine.py
OCR reading of PDF or image files
"""

import io
import numpy as np
from PIL import Image
import easyocr
from pathlib import Path

_reader: easyocr.Reader | None = None

def get_reader(gpu: bool = False) -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("[OCR] Initializing reader...")
        _reader = easyocr.Reader(["fr"], gpu=gpu)
    return _reader

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def _pdf_pages_to_pil(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    try:
        import pydfium2 as pdfium
    except ImportError as e:
        raise ImportError(
            "pydfium2 is required to read PDF files. Please install it with `pip install pydfium2`."
        ) from e
    
    scale = dpi / 72.0
    doc = pdfium.PdfDocument(str(pdf_path))
    pages = []
    for page in doc:
        bitmap = page.render(scale=scale, rotation=0)
        pil_img = bitmap.to_pil()
        pages.append(pil_img)
        page.close()

    doc.close()
    return pages

def _combine_pages(pages: list[Image.Image]) -> Image.Image:
    if len(pages) == 1:
        return pages[0]
    
    width = max(p.width for p in pages)
    total_height = sum(p.height for p in pages)
    combined = Image.new("RBG", (width, total_height), (255, 255, 255))
    y = 0
    for page in pages:
        combined.paste(page, (0, y))
        y += page.height
    return combined

def run_ocr_on_file(file_path: Path, dpi: int = 200, gpu: bool = False) -> str:
    suffix = file_path.suffix.lower()
    reader = get_reader(gpu=gpu)

    if suffix == ".pdf":
        print(f" [PDF->OCR] {file_path.name}")
        pages = _pdf_pages_to_pil(file_path, dpi=dpi)
        image = _combine_pages(pages)
        arr = _pil_to_numpy(image)

    elif suffix in {".png", ".jpg", ".jpeg"}:
        print(f" [IMG->OCR] {file_path.name}")
        image = Image.open(file_path)
        arr = _pil_to_numpy(image)

    else:
        raise ValueError(f"Format non supporté: {suffix}")
    
    results = reader.readtext(arr, detail=0)
    return "\n".join(results)