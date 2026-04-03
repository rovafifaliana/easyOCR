"""
pipeline.py
-----------
Principal pipeline
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports locaux
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ocr_engine import run_ocr_on_file
from extractor import extract_ot, extract_company

# ---------------------------------------------------------------------------
# Extensions supportées
# ---------------------------------------------------------------------------
PDF_EXTS = {".pdf", ".PDF"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
ALL_EXTS  = PDF_EXTS | IMG_EXTS


def collect_files(input_dir: Path) -> list[Path]:
    """Retourne tous les fichiers supportés (PDF + images) dans input_dir."""
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALL_EXTS
    )
    return files


def process_ot(file_path: Path, output_dir: Path, dpi: int, gpu: bool, skip_existing: bool) -> None:
    """Traite un fichier OT : OCR + extraction → JSON."""
    doc_id   = file_path.stem
    out_path = output_dir / f"{doc_id}.json"

    if skip_existing and out_path.exists():
        print(f"  [SKIP] {doc_id} (déjà traité)")
        return

    print(f"\n[OT] Traitement : {file_path.name}")

    # 1. OCR
    text = run_ocr_on_file(file_path, dpi=dpi, gpu=gpu)

    # 2. Extraction LLM (directement depuis le texte, pas de JSON intermédiaire)
    result = extract_ot(text, doc_id)

    # 3. Écriture JSON final
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  [OK] → {out_path.name}")


def process_company_folder(
    folder_path: Path,
    output_dir: Path,
    dpi: int,
    gpu: bool,
    skip_existing: bool,
) -> None:
    """
    Traite un dossier de statuts de société.
    Chaque sous-dossier = une société ; son nom = identifiant unique.
    """
    doc_id   = folder_path.name
    out_path = output_dir / f"{doc_id}.json"

    if skip_existing and out_path.exists():
        print(f"  [SKIP] {doc_id} (déjà traité)")
        return

    print(f"\n[COMPANY] Traitement : {folder_path.name}")

    items       = []
    doc_inserted = ""

    image_files = sorted(
        f for f in folder_path.rglob("*")
        if f.is_file() and f.suffix.lower() in ALL_EXTS
    )

    for i, img_path in enumerate(image_files):
        # Détection pièce d'identité (pour le champ piece_identite)
        name_upper = img_path.name.upper()
        if name_upper.startswith("CARTE_IDENTITE"):
            doc_inserted = "CARTE D'IDENTITE NATIONALE"
            continue
        if name_upper.startswith("CARTE_RESID"):
            doc_inserted = "CARTE DE RESIDENT"
            continue

        text = run_ocr_on_file(img_path, dpi=dpi, gpu=gpu)
        items.append({"id": f"{doc_id}_{i}", "text": text})

    if not items:
        print(f"  [WARN] Aucun fichier traitable dans {folder_path}")
        return

    result = extract_company(items, doc_id, doc_inserted)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"  [OK] → {out_path.name}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline OCR + extraction LLM")
    parser.add_argument("--input",  default="input",  help="Dossier d'entrée (PDF/images)")
    parser.add_argument("--output", default="output", help="Dossier de sortie (JSON)")
    parser.add_argument("--type",   default="ot",     choices=["ot", "company"],
                        help="Type de document : ot (ordre de transfert) ou company (statuts)")
    parser.add_argument("--gpu",    action="store_true", help="Utiliser le GPU pour EasyOCR")
    parser.add_argument("--dpi",    type=int, default=200, help="DPI pour la conversion PDF")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Ignorer les fichiers dont le JSON de sortie existe déjà")
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        print(f"[ERREUR] Dossier d'entrée introuvable : {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Pipeline OCR ===")
    print(f"  Type        : {args.type}")
    print(f"  Entrée      : {input_dir}")
    print(f"  Sortie      : {output_dir}")
    print(f"  DPI         : {args.dpi}")
    print(f"  GPU         : {args.gpu}")
    print(f"  Skip existants : {args.skip_existing}")
    print()

    t_start = time.time()
    ok = err = 0

    if args.type == "ot":
        files = collect_files(input_dir)
        if not files:
            print("[WARN] Aucun fichier PDF/image trouvé dans le dossier d'entrée.")
            return

        print(f"{len(files)} fichier(s) à traiter.\n")
        for file_path in files:
            try:
                process_ot(file_path, output_dir, args.dpi, args.gpu, args.skip_existing)
                ok += 1
            except Exception as e:
                print(f"  [ERREUR] {file_path.name} : {e}")
                err += 1

    elif args.type == "company":
        # Pour company : chaque sous-dossier = une société
        subfolders = sorted(p for p in input_dir.iterdir() if p.is_dir())
        if not subfolders:
            print("[WARN] Aucun sous-dossier trouvé dans le dossier d'entrée.")
            return

        print(f"{len(subfolders)} dossier(s) de société à traiter.\n")
        for folder in subfolders:
            try:
                process_company_folder(folder, output_dir, args.dpi, args.gpu, args.skip_existing)
                ok += 1
            except Exception as e:
                print(f"  [ERREUR] {folder.name} : {e}")
                err += 1

    elapsed = time.time() - t_start
    print(f"\n=== Terminé en {elapsed:.1f}s : {ok} succès, {err} erreur(s) ===")


if __name__ == "__main__":
    main()