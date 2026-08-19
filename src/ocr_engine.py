"""
ocr_engine.py
OCR reading of PDF or image files
"""

import io
import re
import unicodedata
import numpy as np
from PIL import Image
import easyocr
from pathlib import Path
from pdf2image import convert_from_path, pdfinfo_from_path

_reader: easyocr.Reader | None = None

STOP_KEYWORDS = [
    "ORDRE DE VIREMENT OCCASIONNEL",
    "VERS L'ETRANGER"
]

def get_reader(gpu: bool = False) -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("[OCR] Initializing reader...")
        _reader = easyocr.Reader(["fr"], gpu=gpu)
    return _reader

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def _normalize(text: str) -> str:
    """Majuscules, sans accents, espaces multiples réduits - pour un matching robuste"""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.upper()
    text = re.sub(r"\s", " ", text)
    return text

def _contains_stop_keyword(lines: list[str]) -> bool:
    joined = _normalize(" ".join(lines))
    return any(_normalize(kw) in joined for kw in STOP_KEYWORDS)

def _get_pdf_page_count(pdf_path: Path) -> int:
    info = pdfinfo_from_path(pdf_path)
    return info["Pages"]

def _pdf_pages_to_pil(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "pdf2image is required to read PDF files. Install with `pip install pdf2image`."
        ) from e

    try:
        pages = convert_from_path(
            pdf_path,
            dpi=dpi
        )
    except Exception as e:
        raise RuntimeError(
            "Erreur lors de la conversion du PDF. Assure-toi que Poppler est installé."
        ) from e

    return pages

def _combine_pages(pages: list[Image.Image]) -> Image.Image:
    if len(pages) == 1:
        return pages[0]
    
    width = max(p.width for p in pages)
    total_height = sum(p.height for p in pages)
    combined = Image.new("RGB", (width, total_height), (255, 255, 255))
    y = 0
    for page in pages:
        combined.paste(page, (0, y))
        y += page.height
    return combined

def _ocr_pdf_until_stop(
        pdf_path: Path,
        reader: easyocr.Reader,
        dpi: int,
        is_stopword: bool
) -> str:
    try:
        num_pages = _get_pdf_page_count(pdf_path)
    except Exception as e:
        raise RuntimeError(
            "Erreur lors de la lecture du PDF."
        ) from e
    all_text_parts: list[str] = []

    for page_num in range(1, num_pages + 1):
        try:
            pages = convert_from_path(
                pdf_path, dpi=dpi, first_page=page_num, last_page=page_num
            )
        except Exception as e:
            raise RuntimeError(
                f"Erreur lors de la conversion de la page {page_num}."
            ) from e
        page_img = pages[0]
        arr = _pil_to_numpy(page_img)
        page_lines = reader.readtext(arr, detail=0)
        page_text = "\n".join(page_lines)

        all_text_parts.append(f"--- Page {page_num} ---\n{page_text}")

        print(f"    [Page {page_num}/{num_pages}] OCR terminé "
              f"({len(page_lines)} lignes détectées)")

        if is_stopword and _contains_stop_keyword(page_lines):
            print(f"    [STOP] 'ordre de virement' détecté page {page_num} "
                  f"-> arrêt de la lecture du PDF.")
            break

    return "\n\n".join(all_text_parts)

def run_ocr_on_file(
        file_path: Path, 
        dpi: int = 200, 
        gpu: bool = False,
        is_stopword: bool = True
) -> str:
    suffix = file_path.suffix.lower()
    reader = get_reader(gpu=gpu)

    if suffix == ".pdf":
        print(f" [PDF->OCR] {file_path.name}")
        # pages = _pdf_pages_to_pil(file_path, dpi=dpi)
        # image = _combine_pages(pages)
        # arr = _pil_to_numpy(image)
        return _ocr_pdf_until_stop(file_path, reader, dpi, is_stopword)

    elif suffix in {".png", ".jpg", ".jpeg"}:
        print(f" [IMG->OCR] {file_path.name}")
        image = Image.open(file_path)
        arr = _pil_to_numpy(image)
        results = reader.readtext(arr, detail=0)
        return "\n".join(results)

    else:
        raise ValueError(f"Format non supporté: {suffix}")
    
    # results = reader.readtext(arr, detail=0)
    # return "\n".join(results)
