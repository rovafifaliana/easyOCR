"""
extract_ot_w_re.py
Extraction structurée d'un ordre de virement occasionnel (OT) à partir du
texte OCR brut, en 100% regex
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
    pour la recherche de labels — ne modifie jamais la valeur extraite.

    les apostrophes/tirets/underscores/slashs sont remplacés par un
    espace AVANT normalisation
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[''`\-_/]", " ", s)
    s = s.upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s

_ALL_LABELS = [
    "DATE", "TYPE ORDRE",

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

    "ADRESSE DU BENEFICIAIRE", "NOM DU BENEFICIAIRE", "ADRESSE",
    "CARACTERISTIQUES DE L ORDRE", "CARACTERISTIQUES DE FORDRE",
    "INSTRUCTIONS COURS", "DEVISE DE L OPERATION",
    "DEVISE DE TRANSFERT", "MONTANT DE L OPERATION EN LETTRE",
]


def _is_probably_label(norm_line: str) -> bool:
    """Heuristique : une ligne qui ressemble à un label de formulaire (donc
    PAS une valeur), pour éviter de capturer le label suivant comme valeur."""
    return any(norm_line.startswith(lbl) for lbl in _ALL_LABELS)


def _looks_like_explanatory_note(s: str) -> bool:
    """Notes entre parenthèses/accolades du formulaire : ce ne sont jamais des valeurs, mais
    l'OCR les coupe parfois sur plusieurs lignes. On les ignore comme
    candidates plutôt que de les renvoyer comme fausse valeur."""
    return bool(re.match(r"^[({]", s)) or bool(re.search(r"[)}]$", s))


_label_regex_cache: dict[str, "re.Pattern[str]"] = {}


def _label_regex(label: str) -> "re.Pattern[str]":
    """Regex à frontières de mots pour un label. IMPORTANT : quand on
    fusionne des lignes pour repérer un label coupé par l'OCR, un
    simple test de sous-chaîne (`in`) peut matcher accidentellement à
    cheval sur deux mots collés par la fusion."""
    if label not in _label_regex_cache:
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in label.split()) + r"\b"
        _label_regex_cache[label] = re.compile(pattern)
    return _label_regex_cache[label]


def _label_match_span(
    norm_lines: list[str], i: int, label_variants: list[str], max_span: int = 5
) -> int:
    """l'OCR coupe parfois un label sur plusieurs lignes physiques"""
    for span in range(1, max_span + 1):
        if i + span > len(norm_lines):
            break
        merged = " ".join(norm_lines[i : i + span]).strip()
        if any(_label_regex(lv).search(merged) for lv in label_variants):
            return span
    return 0


def _find_label_positions(norm_lines: list[str], label_variants: list[str], max_span: int = 5):
    """Génère (index_ligne, span) pour chaque occurrence d'un label, y
    compris les labels coupés sur plusieurs lignes par l'OCR (voir
    _label_match_span)."""
    for i in range(len(norm_lines)):
        span = _label_match_span(norm_lines, i, label_variants, max_span)
        if span:
            yield i, span


def _extract_after_label(
    lines: list[str],
    norm_lines: list[str],
    label_variants: list[str],
    window: int = 3,
    same_line_sep: str = ":",
    multiline: bool = False,
) -> Optional[str]:
    """Cherche un label (éventuellement coupé sur plusieurs lignes par
    l'OCR)
    """
    for i, span in _find_label_positions(norm_lines, label_variants):
        # 1. même ligne après le séparateur (on ne teste que la 1ère ligne
        #    du label : c'est là qu'un ':' aurait du sens)
        raw_line = lines[i]
        if same_line_sep in raw_line:
            after = raw_line.split(same_line_sep, 1)[1].strip(" :\t")
            if after and not _is_probably_label(_normalize(after)) and not _looks_like_explanatory_note(after):
                return after

        # 2. lignes suivant la fin du label (i + span)
        collected: list[str] = []
        start = i + span
        for j in range(start, min(start + window, len(lines))):
            candidate = lines[j].strip()
            if not candidate:
                if collected:
                    break  # ligne vide après avoir commencé à collecter -> fin de la valeur
                continue
            if _is_probably_label(norm_lines[j]):
                break
            if _looks_like_explanatory_note(candidate):
                continue
            collected.append(candidate)
            if not multiline:
                break  # comportement historique : une seule ligne suffit
        if collected:
            return " - ".join(collected)

    return None

# --------------------------------------------------------------------------- #
# Extraction de champs à format fixe
# --------------------------------------------------------------------------- #
_IBAN_RE = re.compile(r"\b([A-Z]{2}[ ]?\d{2}(?:[ ]?[A-Z0-9]{2,4}){3,8})\b")
_BIC_RE = re.compile(r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b")

_CURRENCY_WORDS = {
    "EUR": "EUR", "EURO": "EUR", "EUROS": "EUR",
    "USD": "USD", "DOLLAR": "USD", "DOLLARS": "USD",
    "MGA": "MGA", "ARIARY": "MGA",
}
_CURRENCY_RE = re.compile(r"\b(EUR|EURO|EUROS|USD|MGA|ARIARY)\b", re.IGNORECASE)

# NB : [\d .]* accepte aussi bien "28 630,23" (milliers séparés par espace)
# que "2588.28" (4 chiffres sans séparateur avant la décimale)
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
    for i, span in _find_label_positions(
        norm_lines, ("MONTANT DE L OPERATION", "MONTANT DE"), max_span=1
    ):
        for j in range(i, min(i + span + 2, len(lines))):
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

def _extract_devise(lines: list[str], norm_lines: list[str], label_variants: list[str]) -> Optional[str]:
    for i, span in _find_label_positions(norm_lines, label_variants):
        for j in range(i, min(i + span + 2, len(lines))):
            m = _CURRENCY_RE.search(lines[j])
            if m:
                return _CURRENCY_WORDS.get(m.group(1).upper(), m.group(1).upper())
    return None

def _extract_devise_any(lines: list[str]) -> Optional[str]:
    # repli global : première devise mentionnée n'importe où dans le document
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
    for i, span in _find_label_positions(norm_lines, ["IBAN", "NUMERO DE COMPTE DU BENEFICIAIRE"]):
        for j in range(i, min(i + span + 2, len(lines))):
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
    # 8 lettres majuscules matcherait le pattern par hasard
    for i, span in _find_label_positions(norm_lines, ["BIC", "SWIFT"]):
        for j in range(i, min(i + span + 2, len(lines))):
            m = _BIC_RE.search(lines[j])
            if m:
                return m.group(1)
    return None


def _extract_date(text: str) -> Optional[str]:
    m = _DATE_RE.search(text)
    return m.group(0).strip() if m else None


def _extract_account_number(lines: list[str], norm_lines: list[str], label_variants: list[str]) -> Optional[str]:
    for i, span in _find_label_positions(norm_lines, label_variants):
        # cherche le motif de compte sur la ligne elle-même puis les suivantes
        for j in range(i, min(i + span + 2, len(lines))):
            m = _ACCOUNT_RE.search(lines[j])
            if m:
                return re.sub(r"\s+", " ", m.group(0)).strip()
    return None


# --------------------------------------------------------------------------- #
# Validation légère (montant en lettres <-> montant en chiffres, champs
# critiques manquants). N'invente jamais de valeur : sert uniquement à
# signaler qu'une revue humaine est nécessaire.
# --------------------------------------------------------------------------- #

_FR_UNITS = {
    "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
    "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
}
_FR_SCALES = {
    "mille": 1000, "million": 1_000_000, "millions": 1_000_000,
    "milliard": 1_000_000_000, "milliards": 1_000_000_000,
}


def _french_words_to_int(text: str) -> Optional[int]:
    """Convertit un montant écrit en toutes lettres (français) en entier.
    Gère les cas standards rencontrés sur les ordres de virement : mille,
    cent, "quatre-vingt" (multiplicatif), "soixante-dix" (additif), avec ou
    sans traits d'union. Retourne None si le texte ne contient aucun mot de
    nombre reconnu -> à traiter comme "non vérifiable", jamais comme "faux".
    """
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[-']", " ", t)
    t = re.sub(r"\bet\b", " ", t)
    tokens = [tok for tok in t.split() if tok not in ("de",)]

    total = 0
    current = 0
    prev = None
    matched_any = False
    for tok in tokens:
        if tok in _FR_UNITS:
            current += _FR_UNITS[tok]
            matched_any = True
        elif tok in ("vingt", "vingts"):
            if prev == "quatre":
                current = current - 4 + 4 * 20
            else:
                current += 20
            matched_any = True
        elif tok in ("cent", "cents"):
            multiplier = current if current != 0 else 1
            current = multiplier * 100
            matched_any = True
        elif tok in _FR_SCALES:
            multiplier = current if current != 0 else 1
            total += multiplier * _FR_SCALES[tok]
            current = 0
            matched_any = True
        prev = tok
    total += current
    return total if matched_any else None


def _validate(result: dict) -> dict:
    """Ajoute un bloc `_validation` au résultat : ne modifie AUCUN champ
    extrait, se contente de signaler les incohérences ou absences sur les
    champs jugés critiques, pour permettre un routage vers une revue
    humaine plutôt qu'un envoi silencieux d'un JSON incomplet/faux."""
    warnings: list[str] = []

    montant = result["caracteristiques_ordre"]["montant_operation"]
    montant_lettres = result["caracteristiques_ordre"]["montant_operation_en_lettre"]
    if montant is not None and montant_lettres:
        parsed_lettres = _french_words_to_int(montant_lettres)
        if parsed_lettres is not None and parsed_lettres != int(round(montant)):
            warnings.append(
                f"incoherence_montant: chiffres={montant} vs lettres='{montant_lettres}' (={parsed_lettres})"
            )

    # critical_fields = {
    #     "caracteristiques_ordre.montant_operation": montant,
    #     "beneficiaire.nom_beneficiaire": result["beneficiaire"]["nom_beneficiaire"],
    #     "banque_beneficiaire.iban_ou_numero_compte_beneficiaire":
    #         result["banque_beneficiaire"]["iban_ou_numero_compte_beneficiaire"],
    #     "banque_beneficiaire.code_bic_swift": result["banque_beneficiaire"]["code_bic_swift"],
    #     "donneur_ordre.numero_compte_a_debiter": result["donneur_ordre"]["numero_compte_a_debiter"],
    # }
    # champs_manquants = [k for k, v in critical_fields.items() if v is None]

    # result["_validation"] = {
    #     "champs_manquants": champs_manquants,
    #     "warnings": warnings,
    #     "a_revoir": bool(champs_manquants) or bool(warnings),
    # }
    return result


# --------------------------------------------------------------------------- #
# Fonction principale
# --------------------------------------------------------------------------- #

def extract_ot_using_regex(text: str, doc_id: str) -> dict:
    """Extraction 100% regex d'un ordre de virement occasionnel.s
    """
    lines = [l.strip() for l in text.split("\n")]
    norm_lines = [_normalize(l) for l in lines]

    montant_operation = _extract_montant(lines, norm_lines)
    devise_operation = _extract_devise(
        lines, norm_lines, ["DEVISE OPERATION", "DEVISE DE L OPERATION", "DEVISE DE LOPERATION"]
    )
    devise_transfert = _extract_devise(
        lines, norm_lines, ["DEVISE DE TRANSFERT"]
    )

    devise_fallback = _extract_devise_any(lines)

    result = {
        "doc_id": doc_id,
        "type_document": "ORDRE DE VIREMENT OCCASIONNEL VERS L'ETRANGER",
        "date": _extract_date(text),
        "type_ordre": _extract_after_label(lines, norm_lines, ["TYPE ORDRE"]),

        "donneur_ordre": {
            "numero_compte_a_debiter": _extract_account_number(
                lines, norm_lines,
                ["NUMERO DE COMPTE A DEBITER", "NUMERO DE COMPTE", "COMPTE A DEBITER"],
            ),
            "noms_prenoms_ou_raison_sociale": _extract_after_label(
                lines, norm_lines,
                ["NOMS ET PRENOMS OU RAISON SOCIALE", "NOM ET PRENOM", "RAISON SOCIALE"],
            ),
            "adresse_donneur_ordre": _extract_after_label(
                lines, norm_lines, ["ADRESSE DU DONNEUR D ORDRE", "ADRESSE DU DONNEUR"],
                multiline=True,
            ),
        },

        "motif_transfert": _extract_after_label(
            lines, norm_lines, ["MOTIF DU TRANSFERT"], window=4, multiline=True,
        ),

        "paiement_frais": {
            "paiement_frais_par": _extract_after_label(
                lines, norm_lines, ["PAIEMENT FRAIS PAR"],
            ),
            "numero_compte_prelevement_frais": _extract_account_number(
                lines, norm_lines,
                ["NUMERO DE COMPTE DE PRELEVEMENT DE FRAIS", "NUMERO DE COMPTE DU PRELEVEMENT DE FRAIS"],
            ),
        },

        "caracteristiques_ordre": {
            "montant_operation": montant_operation,
            "devise_operation": devise_operation or devise_fallback,
            "montant_operation_en_lettre": _extract_after_label(
                lines, norm_lines, ["MONTANT DE L OPERATION EN LETTRE", "MONTANT DE OPERATION EN LETTRE"],
            ),
            "devise_transfert": devise_transfert or devise_fallback,
            "instructions_cours": _extract_after_label(
                lines, norm_lines, ["INSTRUCTIONS COURS", "INSTRUCTIONS"],
            ),
            "numero_dom": _extract_after_label(lines, norm_lines, ["NUMERO DOM"]),
        },

        "beneficiaire": {
            "nom_beneficiaire": _extract_after_label(lines, norm_lines, ["NOM DU BENEFICIAIRE"]),
            "adresse_beneficiaire": _extract_after_label(
                lines, norm_lines, ["ADRESSE DU BENEFICIAIRE"], multiline=True,
            ),
            "pays_beneficiaire": _extract_after_label(
                lines, norm_lines, ["PAYS DU BENEFICIAIRE"],
            ),
        },

        "banque_beneficiaire": {
            "iban_ou_numero_compte_beneficiaire": _extract_iban(lines, norm_lines),
            "code_bic_swift": _extract_bic(lines, norm_lines),
            "nom_banque_beneficiaire": _extract_after_label(
                lines, norm_lines, ["NOM DE LA BANQUE DU BENEFICIAIRE"],
            ),
            "adresse_banque_beneficiaire": _extract_after_label(
                lines, norm_lines, ["ADRESSE DE LA BANQUE DU BENEFICIAIRE"], multiline=True,
            ),
            "pays_banque_beneficiaire": _extract_after_label(
                lines, norm_lines, ["PAYS DE LA BANQUE DU BENEFICIAIRE"],
            ),
        },
    }
    return _validate(result)