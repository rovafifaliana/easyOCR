#!/usr/bin/env bash
# =============================================================================
# run.sh — Point d'entrée unique du pipeline OCR
# =============================================================================
# Usage :
#   ./run.sh [OPTIONS]
#
# Options :
#   --type      ot|company          Type de document (défaut : ot)
#   --input     <chemin>            Dossier contenant les PDF/images (défaut : ./input)
#   --output    <chemin>            Dossier de sortie JSON (défaut : ./output)
#   --dpi       <entier>            Résolution pour PDF→image en mémoire (défaut : 200)
#   --gpu                           Activer le GPU pour EasyOCR
#   --skip-existing                 Ne pas retraiter les fichiers déjà convertis
#   --install                       Installer les dépendances Python avant de lancer
#   --help                          Afficher cette aide
# =============================================================================

set -euo pipefail

# ---------- Chemins relatifs au script ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
INPUT_DIR="$SCRIPT_DIR/input"
OUTPUT_DIR="$SCRIPT_DIR/output"

# ---------- Valeurs par défaut ------------------------------------------------
DOC_TYPE="ot"
DPI=200
GPU_FLAG=""
SKIP_FLAG=""
DO_INSTALL=0

# ---------- Parsing des arguments ---------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)           DOC_TYPE="$2";    shift 2 ;;
    --input)          INPUT_DIR="$2";   shift 2 ;;
    --output)         OUTPUT_DIR="$2";  shift 2 ;;
    --dpi)            DPI="$2";         shift 2 ;;
    --gpu)            GPU_FLAG="--gpu"; shift   ;;
    --skip-existing)  SKIP_FLAG="--skip-existing"; shift ;;
    --install)        DO_INSTALL=1;     shift   ;;
    --help|-h)
      sed -n '2,25p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "[ERREUR] Argument inconnu : $1"
      exit 1
      ;;
  esac
done

# ---------- Démarrage d'Ollama (si pas déjà en route) ------------------------
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "[INFO] Démarrage d'Ollama en arrière-plan..."
  ollama serve &
  OLLAMA_PID=$!
  echo "[INFO] Ollama PID : $OLLAMA_PID"
  sleep 15
else
  echo "[INFO] Ollama déjà en cours d'exécution."
  OLLAMA_PID=""
fi

# ---------- Création des dossiers output si absents --------------------
mkdir -p "$OUTPUT_DIR"

# ---------- Lancement du pipeline Python -------------------------------------
echo ""
echo "=== Lancement du pipeline OCR ==="
python3 "$SRC_DIR/pipeline.py" \
  --type    "$DOC_TYPE"   \
  --input   "$INPUT_DIR"  \
  --output  "$OUTPUT_DIR" \
  --dpi     "$DPI"        \
  $GPU_FLAG               \
  $SKIP_FLAG

EXIT_CODE=$?

exit $EXIT_CODE
