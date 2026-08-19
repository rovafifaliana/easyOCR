"""
extract_ot_w_re.py
Extraction structurée d'un ordre de virement occasionnel (OT) à partir du
texte OCR brut, en 100% regex (pas de LLM — Mistral/Ollama écarté faute
de ressources).

Usage :
    result = extract_ot(text, doc_id)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# --------------------------------------------------------------------------- #
# Utilitaires de normalisation / recherche de labels
# --------------------------------------------------------------------------- #

def _normalize(s: str) -> str:
    """Majuscule, sans accents, espaces multiples réduits. Utilisé UNIQUEMENT
    pour la recherche de labels — ne modifie jamais la valeur extraite."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Labels connus du formulaire BNI (phrases complètes, pas des mots isolés,
# pour ne pas filtrer à tort une vraie valeur qui commencerait par un mot
# comme "Nom" ou "Ville"). On teste un "startswith" SANS limite de longueur,
# car l'OCR fusionne parfois label + valeur sur une même ligne
# (ex: "Adresse de la banque du beneficiaire : The Mauritius Commercial...").
_ALL_LABELS = [
    "DATE", "TYPE ORDRE", "DONNEUR D ORDRE",
    "NUMERO DE COMPTE A DEBITER", "NUMERO DE COMPTE",
    "NOMS ET PRENOMS OU RAISON SOCIALE", "RAISON SOCIALE",
    "ADRESSE DU DONNEUR D ORDRE", "MOTIF DU TRANSFERT",
    "MODALITE DE PAIEMENT DES FRAIS", "PAIEMENT FRAIS PAR",
    "CARACTERISTIQUES DE", "MONTANT DE L OPERATION", "MONTANT DE",
    "DEVISE OPERATION", "DEVISE DE TRANSFERT", "DEVISE",
    "INSTRUCTIONS", "NUMERO DOM", "BENEFICIAIRE", "NOM DU BENEFICIAIRE",
    "SERVICE", "SOUS SERVICE", "NUMERO ET NOM DE RUE", "NUMERO DE BATIMENT",
    "NOM DE BATIMENT", "ETAGE", "BOITE POSTALE", "CODE POSTALE",
    "NUMERO DE PORTE", "VILLE", "LOCALITE", "NOM DU DISTRICT",
    "REGION OU ETAT OU COMTE", "PAYS DU BENEFICIAIRE", "BANQUE BENEFICIAIRE",
    "IBAN OU NUMERO DE COMPTE DU BENEFICIAIRE", "CODE BIC",
    "NOM DE LA BANQUE DU BENEFICIAIRE", "ADRESSE DE LA BANQUE DU BENEFICIAIRE",
    "PAYS DE LA BANQUE DU BENEFICIAIRE", "PAYS", "SIGNATURE",
]


def _is_probably_label(norm_line: str) -> bool:
    """Heuristique : une ligne qui ressemble à un label de formulaire (donc
    PAS une valeur), pour éviter de capturer le label suivant comme valeur."""
    return any(norm_line.startswith(lbl) for lbl in _ALL_LABELS)


def _extract_after_label(
    lines: list[str],
    norm_lines: list[str],
    label_variants: list[str],
    window: int = 3,
    same_line_sep: str = ":",
) -> Optional[str]:
    """Cherche une ligne contenant un des labels, puis retourne :
    1. le reste de la même ligne après ':' si présent et non vide,
    2. sinon la première ligne non vide dans les `window` lignes suivantes
       qui ne ressemble pas elle-même à un label.
    """
    for i, norm_line in enumerate(norm_lines):
        if not any(lv in norm_line for lv in label_variants):
            continue

        # 1. même ligne après le séparateur
        raw_line = lines[i]
        if same_line_sep in raw_line:
            after = raw_line.split(same_line_sep, 1)[1].strip(" :\t")
            if after and not _is_probably_label(_normalize(after)):
                return after

        # 2. lignes suivantes
        for j in range(i + 1, min(i + 1 + window, len(lines))):
            candidate = lines[j].strip()
            if not candidate:
                continue
            if _is_probably_label(norm_lines[j]):
                continue
            return candidate

    return None


# --------------------------------------------------------------------------- #
# Extraction de champs à format fixe
# --------------------------------------------------------------------------- #

# NB : ces regex sont appliquées LIGNE PAR LIGNE (jamais sur le texte
# entier avec \n) pour éviter qu'un \s permissif n'avale le mot ou le
# label de la ligne suivante (bug observé : IBAN "avalant" le mot "CODE"
# de la ligne suivante, ou un BIC fusionné avec l'IBAN).
_IBAN_RE = re.compile(r"\b([A-Z]{2}[ ]?\d{2}(?:[ ]?[A-Z0-9]{2,4}){3,8})\b")
_BIC_RE = re.compile(r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b")

_CURRENCY_WORDS = {
    "EUR": "EUR", "EURO": "EUR", "EUROS": "EUR",
    "USD": "USD", "DOLLAR": "USD", "DOLLARS": "USD",
    "MGA": "MGA", "ARIARY": "MGA",
}
_CURRENCY_RE = re.compile(r"\b(EUR|EURO|EUROS|USD|MGA|ARIARY)\b", re.IGNORECASE)

# NB : [\d .]* accepte aussi bien "28 630,23" (milliers séparés par espace)
# que "2588.28" (4 chiffres sans séparateur avant la décimale) — un regex
# figé sur des groupes de 3 chiffres cassait ce second cas.
_NUMBER_RE = re.compile(r"\d[\d .]*(?:[.,]\d{1,2})?")

_MONTANT_RE = re.compile(
    r"(\d[\d .]*(?:[.,]\d{1,2})?)\s*(EUR|EURO|EUROS|USD|MGA|ARIARY)",
    re.IGNORECASE,
)
# Certains formulaires écrivent la devise AVANT le montant ("EUR 28 630.23")
_MONTANT_CURRENCY_FIRST_RE = re.compile(
    r"(EUR|EURO|EUROS|USD|MGA|ARIARY)\s*(\d[\d .]*(?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b(\d{1,2})\s*(?:/|-|\s)\s*"
    r"(\d{1,2}|JANVIER|FEVRIER|MARS|AVRIL|MAI|JUIN|JUILLET|AOUT|"
    r"SEPTEMBRE|OCTOBRE|NOVEMBRE|DECEMBRE)\s*(?:/|-|\s)\s*(\d{4})\b",
    re.IGNORECASE,
)

# Numéro de compte : préfixes bancaires observés "00005", "00007" suivis
# d'une longue suite de chiffres, éventuellement séparés par espaces / slashs.
_ACCOUNT_RE = re.compile(r"\b0000[0-9]\s?[/\s]?\s?\d[\d\s/]{9,25}\d\b")


def _clean_iban(raw: str) -> str:
    return re.sub(r"\s+", "", raw).upper()


def _parse_montant(raw: str) -> Optional[float]:
    cleaned = raw.replace(" ", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_montant(lines: list[str], norm_lines: list[str]) -> Optional[float]:
    # 1. essai via un label précis "Montant de l'operation" -> premier
    #    nombre trouvé sur la ligne du label ou les 2 suivantes. On
    #    n'accepte pas n'importe quel texte (ex: "chiffres)") comme valeur :
    #    on cherche directement un motif numérique dans la fenêtre.
    for i, norm_line in enumerate(norm_lines):
        if not any(lv in norm_line for lv in ("MONTANT DE L OPERATION", "MONTANT DE")):
            continue
        for j in range(i, min(i + 3, len(lines))):
            m = _NUMBER_RE.search(lines[j])
            if m:
                parsed = _parse_montant(m.group(0))
                if parsed is not None and parsed > 0:
                    return parsed

    # 2. repli : "nombre + devise" ou "devise + nombre" collés sur UNE
    #    SEULE ligne, n'importe où dans le document.
    for line in lines:
        m = _MONTANT_RE.search(line)
        if m:
            parsed = _parse_montant(m.group(1))
            if parsed is not None:
                return parsed
        m = _MONTANT_CURRENCY_FIRST_RE.search(line)
        if m:
            parsed = _parse_montant(m.group(2))
            if parsed is not None:
                return parsed
    return None


def _extract_devise(lines: list[str], norm_lines: list[str]) -> Optional[str]:
    # On cherche une devise VALIDE (EUR/USD/MGA/...) à proximité d'un label
    # "devise" — on n'accepte jamais un texte quelconque comme "chiffres)"
    # même s'il suit le label, pour éviter les faux positifs sur les
    # documents composites (plusieurs formulaires concaténés).
    for i, norm_line in enumerate(norm_lines):
        if not any(lv in norm_line for lv in ("DEVISE DE TRANSFERT", "DEVISE OPERATION", "DEVISE")):
            continue
        for j in range(i, min(i + 3, len(lines))):
            m = _CURRENCY_RE.search(lines[j])
            if m:
                return _CURRENCY_WORDS.get(m.group(1).upper(), m.group(1).upper())

    # repli : première devise mentionnée n'importe où dans le document
    for line in lines:
        m = _CURRENCY_RE.search(line)
        if m:
            return _CURRENCY_WORDS.get(m.group(1).upper(), m.group(1).upper())
    return None


def _extract_iban(lines: list[str], norm_lines: list[str]) -> Optional[str]:
    def _valid(raw: str) -> Optional[str]:
        cleaned = _clean_iban(raw)
        return cleaned if 15 <= len(cleaned) <= 34 else None

    # 1. priorité : lignes proches du label IBAN
    for i, norm_line in enumerate(norm_lines):
        if "IBAN" in norm_line or "NUMERO DE COMPTE DU BENEFICIAIRE" in norm_line:
            for j in range(i, min(i + 3, len(lines))):
                m = _IBAN_RE.search(lines[j])
                if m and (v := _valid(m.group(1))):
                    return v

    # 2. repli : n'importe quelle ligne du document (une ligne à la fois,
    #    jamais le texte entier -> pas de contamination inter-lignes)
    for line in lines:
        m = _IBAN_RE.search(line)
        if m and (v := _valid(m.group(1))):
            return v
    return None


def _extract_bic(lines: list[str], norm_lines: list[str]) -> Optional[str]:
    # On exige la proximité du mot-clé BIC/SWIFT : sans lui, un mot OCR de
    # 8 lettres majuscules matcherait le pattern par hasard (faux positif
    # observé : "VIREMENT"). Sans mot-clé -> on préfère renvoyer None.
    for i, norm_line in enumerate(norm_lines):
        if "BIC" in norm_line or "SWIFT" in norm_line:
            for j in range(i, min(i + 3, len(lines))):
                m = _BIC_RE.search(lines[j])
                if m:
                    return m.group(1)
    return None


def _extract_date(text: str) -> Optional[str]:
    m = _DATE_RE.search(text)
    return m.group(0).strip() if m else None


def _extract_account_number(lines: list[str], norm_lines: list[str], label_variants: list[str]) -> Optional[str]:
    for i, norm_line in enumerate(norm_lines):
        if not any(lv in norm_line for lv in label_variants):
            continue
        # cherche le motif de compte sur la ligne elle-même puis les 2 suivantes
        for j in range(i, min(i + 3, len(lines))):
            m = _ACCOUNT_RE.search(lines[j])
            if m:
                return re.sub(r"\s+", " ", m.group(0)).strip()
    return None


# --------------------------------------------------------------------------- #
# Fonction principale
# --------------------------------------------------------------------------- #

def extract_ot(text: str, doc_id: str) -> dict:
    """Extraction 100% regex d'un ordre de virement occasionnel.

    NB : les champs "flous" (noms, adresses) dépendent fortement de la
    qualité de l'OCR. Sur un scan très dégradé, ils peuvent rester à null
    ou être imprécis — il n'y a pas de correction sémantique sans LLM.
    """
    lines = [l.strip() for l in text.split("\n")]
    norm_lines = [_normalize(l) for l in lines]

    result = {
        "doc_id": doc_id,
        "date": _extract_date(text),
        "motif": _extract_after_label(lines, norm_lines, ["MOTIF DU TRANSFERT"]),
        "montant": _extract_montant(lines, norm_lines),
        "devise": _extract_devise(lines, norm_lines),
        "donneur_ordre": {
            "nom": _extract_after_label(
                lines, norm_lines,
                ["NOMS ET PRENOMS OU RAISON SOCIALE", "NOM ET PRENOM", "RAISON SOCIALE"],
            ),
            "adresse": _extract_after_label(
                lines, norm_lines, ["ADRESSE DU DONNEUR D ORDRE", "ADRESSE DU DONNEUR"],
            ),
            "numero_compte": _extract_account_number(
                lines, norm_lines,
                ["NUMERO DE COMPTE A DEBITER", "NUMERO DE COMPTE", "COMPTE A DEBITER"],
            ),
        },
        "beneficiaire": {
            "nom": _extract_after_label(lines, norm_lines, ["NOM DU BENEFICIAIRE"]),
            "adresse": _extract_after_label(lines, norm_lines, ["ADRESSE DU BENEFICIAIRE"]),
            "ville": _extract_after_label(lines, norm_lines, ["VILLE"]),
            "pays": _extract_after_label(lines, norm_lines, ["PAYS DU BENEFICIAIRE"]),
        },
        "banque_beneficiaire": {
            "nom": _extract_after_label(
                lines, norm_lines, ["NOM DE LA BANQUE DU BENEFICIAIRE"],
            ),
            "adresse": _extract_after_label(
                lines, norm_lines, ["ADRESSE DE LA BANQUE DU BENEFICIAIRE"],
            ),
            "pays": _extract_after_label(
                lines, norm_lines, ["PAYS DE LA BANQUE DU BENEFICIAIRE"],
            ),
            "iban": _extract_iban(lines, norm_lines),
            "bic": _extract_bic(lines, norm_lines),
        },
    }
    return result


# if __name__ == "__main__":
#     import json
#     import sys
#     from pathlib import Path

#     if len(sys.argv) < 2:
#         print("Usage: python extract_ot.py <fichier.json produit par l'OCR>")
#         sys.exit(1)

#     raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
#     out = extract_ot(raw["text"], raw["id"])
#     print(json.dumps(out, ensure_ascii=False, indent=2))