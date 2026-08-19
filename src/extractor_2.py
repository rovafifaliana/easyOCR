"""
extractor_rules.py
Extraction SANS LLM des champs d'un Ordre de Transfert/Virement (OT/OV)
depuis du texte OCR linéarisé, par ancrage sur mots-clés + regex.

100% déterministe, pas de dépendance réseau, rapide.

Les listes de labels ci-dessous sont des points de départ génériques
   (vocabulaire bancaire FR courant). Elles doivent être calibrées avec
   de vrais exemples de tes documents OCR pour être fiables : les noms
   de champs varient d'une banque à l'autre.
"""

import re
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Patterns génériques réutilisables
# ---------------------------------------------------------------------------

SWIFT_RE = r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"
IBAN_RE = r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"
ACCOUNT_RE = r"\d[\d\s]{8,}\d"
AMOUNT_RE = r"\d[\d\s.,]{0,15}\d"
DATE_NUM_RE = r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}"
DATE_LETTRE_RE = r"\d{1,2}\s+\w+\s+\d{4}"

MOIS = {
    "janvier": "01", "fevrier": "02", "février": "02", "mars": "03",
    "avril": "04", "mai": "05", "juin": "06", "juillet": "07",
    "aout": "08", "août": "08", "septembre": "09", "octobre": "10",
    "novembre": "11", "decembre": "12", "décembre": "12",
}


def _normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    m = re.match(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})", raw)
    if m:
        d, mo, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", raw, re.IGNORECASE)
    if m:
        month_num = MOIS.get(m.group(2).lower())
        if month_num:
            return f"{m.group(3)}-{month_num}-{m.group(1).zfill(2)}"
    return None


def _clean_amount(raw: str) -> Optional[float]:
    if not raw:
        return None
    raw = raw.strip()
    # Format "1.234.567,89" (séparateur milliers = point, décimales = virgule)
    if "," in raw and raw.count(".") >= 1:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", ".")
    raw = re.sub(r"[^\d.]", "", raw)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _find_after_label(
    text: str,
    labels: List[str],
    stop_labels: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    max_chars: int = 150,
) -> Optional[str]:
    """
    Cherche le premier label trouvé parmi `labels` et retourne ce qui suit
    (jusqu'à un stop_label ou max_chars).
    Si `pattern` est fourni, retourne le 1er match de ce pattern dans la zone.
    Sinon retourne la première ligne/segment non vide après le label.
    """
    for label in labels:
        m = re.search(re.escape(label), text, re.IGNORECASE)
        if not m:
            continue
        window = text[m.end(): m.end() + max_chars]

        if stop_labels:
            positions = [
                window.lower().find(sl.lower())
                for sl in stop_labels
                if sl.lower() in window.lower()
            ]
            if positions:
                window = window[: min(positions)]

        window = window.strip(" :\n\t-")

        if pattern:
            pm = re.search(pattern, window)
            if pm:
                return pm.group(0).strip()
            continue

        segment = re.split(r"[\n]", window)[0].strip()
        return segment or None

    return None


def _split_adresse(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Sépare une adresse brute en (rue, ville). Heuristique simple :
    la ville est le dernier segment après la dernière virgule."""
    if not raw:
        return None, None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) >= 2:
        return ", ".join(parts[:-1]), parts[-1]
    return raw.strip(), None


# ---------------------------------------------------------------------------
# Extraction principale
# ---------------------------------------------------------------------------

def extract_ot(text: str, doc_id: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError(f"[{doc_id}] Texte OCR vide.")

    # --- type d'ordre ---
    type_ordre = None
    if re.search(r"\bOCCASIONNEL\b", text, re.IGNORECASE):
        type_ordre = "OCCASIONNEL"
    elif re.search(r"\bPERMANENT\b", text, re.IGNORECASE):
        type_ordre = "PERMANENT"

    # --- date ---
    date_raw = _find_after_label(
        text, ["Date", "Fait à", "Fait le", "Le "],
        pattern=f"{DATE_NUM_RE}|{DATE_LETTRE_RE}",
    )
    date = _normalize_date(date_raw) if date_raw else None

    # --- donneur d'ordre ---
    numero_compte_debit = _find_after_label(
        text,
        ["Compte à débiter", "N° compte à débiter", "Compte débit", "N° de compte à débiter"],
        pattern=ACCOUNT_RE,
    )
    numero_compte_frais = _find_after_label(
        text, ["Compte frais", "N° compte frais", "Compte pour frais"],
        pattern=ACCOUNT_RE,
    )
    paiement_frais_par = _find_after_label(
        text, ["Frais payés par", "Paiement des frais par", "Frais à la charge de"],
    )
    nom_raison_sociale = _find_after_label(
        text, ["Nom et raison sociale", "Donneur d'ordre", "Nom du donneur d'ordre"],
        stop_labels=["Compte", "Adresse", "Bénéficiaire"],
    )
    adresse_donneur = _find_after_label(
        text, ["Adresse du donneur d'ordre", "Adresse"],
        stop_labels=["Compte", "Bénéficiaire", "Motif"],
    )
    rue_donneur, ville_donneur = _split_adresse(adresse_donneur)

    # --- transfert ---
    motif = _find_after_label(
        text, ["Motif du transfert", "Motif", "Objet du transfert", "Objet"],
        stop_labels=["Montant", "Devise"],
    )
    montant_raw = _find_after_label(
        text, ["Montant du transfert", "Montant à transférer", "Montant"],
        pattern=AMOUNT_RE,
    )
    montant = _clean_amount(montant_raw) if montant_raw else None
    devise_operation = _find_after_label(
        text, ["Devise de l'opération", "Devise opération"],
        pattern=r"[A-Z]{3}",
    )
    devise_transfert = _find_after_label(
        text, ["Devise du transfert", "Devise"],
        pattern=r"[A-Z]{3}",
    )
    montant_en_lettres = _find_after_label(
        text, ["Montant en lettres", "Arrêté la présente somme de"],
        max_chars=200,
    )
    cours = _find_after_label(text, ["Cours", "Taux de change"], pattern=AMOUNT_RE)
    numero_dom = _find_after_label(text, ["N° Dom", "N° de domiciliation", "Domiciliation"])

    # --- bénéficiaire ---
    nom_beneficiaire = _find_after_label(
        text, ["Nom du bénéficiaire", "Bénéficiaire"],
        stop_labels=["Adresse", "IBAN", "Compte", "Banque"],
    )
    adresse_beneficiaire = _find_after_label(
        text, ["Adresse du bénéficiaire"],
        stop_labels=["IBAN", "Compte", "Banque", "Pays"],
    )
    rue_beneficiaire, ville_beneficiaire = _split_adresse(adresse_beneficiaire)
    pays_beneficiaire = _find_after_label(text, ["Pays du bénéficiaire", "Pays"])

    # --- banque bénéficiaire ---
    iban = _find_after_label(text, ["IBAN"], pattern=IBAN_RE) \
        or _find_after_label(text, ["N° de compte bénéficiaire", "Compte bénéficiaire"], pattern=ACCOUNT_RE)
    swift = _find_after_label(text, ["SWIFT", "BIC", "Code SWIFT", "Code SWIFT/BIC"], pattern=SWIFT_RE)
    if not swift:
        m = re.search(SWIFT_RE, text)
        swift = m.group(0) if m else None
    nom_banque = _find_after_label(
        text, ["Banque du bénéficiaire", "Nom de la banque"],
        stop_labels=["Adresse", "SWIFT", "IBAN"],
    )
    adresse_banque = _find_after_label(
        text, ["Adresse de la banque"], stop_labels=["SWIFT", "IBAN", "Pays"],
    )
    rue_banque, ville_banque = _split_adresse(adresse_banque)
    pays_banque = _find_after_label(text, ["Pays de la banque"])

    return {
        "id": doc_id,
        "date": date,
        "type_ordre": type_ordre,
        "donneur_ordre": {
            "numero_compte_debit": numero_compte_debit,
            "numero_compte_frais": numero_compte_frais,
            "paiement_frais_par": paiement_frais_par,
            "nom_raison_sociale": nom_raison_sociale,
            "rue": rue_donneur,
            "ville": ville_donneur,
        },
        "transfert": {
            "motif": motif,
            "montant": montant,
            "devise_operation": devise_operation or devise_transfert,
            "montant_en_lettres": montant_en_lettres,
            "devise_transfert": devise_transfert,
            "cours": _clean_amount(cours) if cours else None,
            "numero_dom": numero_dom,
        },
        "beneficiaire": {
            "nom": nom_beneficiaire,
            "rue": rue_beneficiaire,
            "ville": ville_beneficiaire,
            "pays": pays_beneficiaire,
        },
        "banque_beneficiaire": {
            "iban_ou_numero_compte": iban,
            "code_bic_swift": swift,
            "nom_banque": nom_banque,
            "rue_banque": rue_banque,
            "ville_banque": ville_banque,
            "pays_banque": pays_banque,
        },
    }


# ---------------------------------------------------------------------------
# Export JSON
# ---------------------------------------------------------------------------

import json
from pathlib import Path
from datetime import datetime, timezone


def save_raw_ocr_json(text: str, doc_id: str, output_path: str, source_file: str = "") -> None:
    """
    Sauvegarde le texte OCR brut dans un JSON, avec un peu de métadonnées.
    Utile pour archiver/déboguer avant extraction.
    """
    payload = {
        "id": doc_id,
        "source_file": source_file,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
    }
    Path(output_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_extraction_json(result: Dict[str, Any], output_path: str) -> None:
    """Sauvegarde le résultat structuré (sortie de extract_ot) dans un JSON."""
    Path(output_path).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    # Petit exemple de test manuel — remplace par un vrai texte OCR
    sample = """
    ORDRE DE VIREMENT OCCASIONNEL
    Date : 12/05/2026
    Nom et raison sociale : SOCIETE EXEMPLE SARL
    Compte à débiter : 5 00001 02105360100 35
    Motif du transfert : MEMBERSHIP FEE 2026
    Montant : 1 250,00
    Devise : EUR
    Bénéficiaire : JOHN DOE ASSOCIATION
    IBAN : BE71096123456769
    SWIFT : JVBABE22
    """
    # 1. Sauver le texte OCR brut (optionnel, pour archivage/debug)
    save_raw_ocr_json(sample, "TEST-001", "ocr_TEST-001.json", source_file="TEST-001.pdf")

    # 2. Extraire les champs structurés
    result = extract_ot(sample, "TEST-001")

    # 3. Sauver le résultat structuré
    save_extraction_json(result, "extraction_TEST-001.json")

    print(json.dumps(result, indent=2, ensure_ascii=False))