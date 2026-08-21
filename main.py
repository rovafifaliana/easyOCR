from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.extract_ot import extract_ot_using_regex

input_folder_path = Path("output_ocr/content/output")
output_folder_path = r"output_final"

def main() -> None:
    stop_int = 5
    files = list(input_folder_path.iterdir())
    for i, file in enumerate(files, start=1):
        print(f"\nTraitement fichier {file.name}")
        with file.open(encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("text", "")
        doc_id = data.get("id", file.stem)

        result = extract_ot_using_regex(text, doc_id)

        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        output = Path(f"{output_folder_path}/res_{doc_id}.json")
        output.write_text(output_json, encoding="utf-8")
        print(f"Résultat écrit dans : {output}")

        i += 1

        if i == stop_int:
            break
        else:
            continue

if __name__ == "__main__":
    main()