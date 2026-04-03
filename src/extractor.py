"""
extractor.py
Extract text from PDF or image files using OCR
"""

import re
import json
import unicodedata
import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_llm_output(raw: str) -> dict:
    """Nettoie la réponse brute du LLM et retourne un dict Python."""
    raw = raw.strip()
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    raw = unicodedata.normalize("NFKC", raw)
    raw = raw.replace("\u201c", '"').replace("\u201d", '"')

    matches = list(re.finditer(r"\{[\s\S]*\}", raw))
    if not matches:
        raise ValueError(f"Aucun JSON trouvé dans la réponse : {raw[:300]}")
    cleaned = matches[-1].group(0)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)   # trailing commas
    return json.loads(cleaned)


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Appelle l'API Ollama et retourne le texte de réponse."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if "response" not in data:
        raise KeyError(f"Clé 'response' absente dans la réponse Ollama : {data}")
    return data["response"]


# ---------------------------------------------------------------------------
# Extraction OT (Ordre de Transfert / Virement)
# ---------------------------------------------------------------------------

def _build_ot_prompt(full_text: str, ot_id: str) -> str:
    return f"""
Tu es un expert en analyse de documents bancaires.
Extrais les informations d'un Ordre de Virement (OV/OT) vers l'étranger.

RÈGLES CRITIQUES :
1. IGNORE : factures, justificatifs, mentions "Banque Centrale", duplicatas.
2. "type_ordre" = type juridique UNIQUEMENT : "OCCASIONNEL" ou "PERMANENT". PAS le motif.
3. "motif" = raison du paiement (ex: "MEMBERSHIP FEE 2026 / Cotisation 2026").
4. "numero_compte_debit" = suite de chiffres (ex: "5 00001 02105360100 35").
5. "paiement_frais_par" = NOM de l'entité qui paie les frais (pas un numéro).
6. "nom_raison_sociale" = nom complet du donneur d'ordre.
7. "code_bic_swift" = code SWIFT/BIC (ex: "JVBABE22"). Cherche dans tout le texte.
8. "montant" = valeur numérique uniquement (ex: 100.00).
9. Pour les adresses : sépare toujours "rue" (numéro + nom de rue) et "ville" (ville + code postal).
10. Si absent ou illisible → null.
11. Retourne UNIQUEMENT un JSON brut valide, sans markdown, sans texte autour.

DOCUMENT :
{full_text}

JSON attendu :
{{
  "id": "{ot_id}",
  "date": null,
  "type_ordre": null,
  "donneur_ordre": {{
    "numero_compte_debit": null,
    "numero_compte_frais": null,
    "paiement_frais_par": null,
    "nom_raison_sociale": null,
    "rue": null,
    "ville": null
  }},
  "transfert": {{
    "motif": null,
    "montant": null,
    "devise_operation": null,
    "montant_en_lettres": null,
    "devise_transfert": null,
    "cours": null,
    "numero_dom": null
  }},
  "beneficiaire": {{
    "nom": null,
    "rue": null,
    "ville": null,
    "pays": null
  }},
  "banque_beneficiaire": {{
    "iban_ou_numero_compte": null,
    "code_bic_swift": null,
    "nom_banque": null,
    "rue_banque": null,
    "ville_banque": null,
    "pays_banque": null
  }}
}}
"""


def _postprocess_ot(result: dict, ot_id: str) -> dict:
    """Corrections et normalisations post-extraction OT."""
    result["id"] = ot_id

    MOIS = {
        "janvier": "01", "fevrier": "02", "février": "02", "mars": "03",
        "avril": "04", "mai": "05", "juin": "06", "juillet": "07",
        "aout": "08", "août": "08", "septembre": "09", "octobre": "10",
        "novembre": "11", "decembre": "12", "décembre": "12",
    }
    date_raw = (result.get("date") or "").strip()
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", date_raw, re.IGNORECASE)
    if m:
        month_num = MOIS.get(m.group(2).lower())
        if month_num:
            result["date"] = f"{m.group(3)}-{month_num}-{m.group(1).zfill(2)}"

    transfert = result.get("transfert", {})

    # Montant → float
    raw_m = transfert.get("montant")
    if raw_m is not None:
        try:
            transfert["montant"] = float(
                re.sub(r"[^\d.]", "", str(raw_m).replace(",", "."))
            )
        except ValueError:
            transfert["montant"] = None

    if not transfert.get("devise_operation") and transfert.get("devise_transfert"):
        transfert["devise_operation"] = transfert["devise_transfert"]

    # type_ordre : si le LLM a mis un motif, corriger
    type_raw = (result.get("type_ordre") or "").lower()
    motif_kw = {"membership", "cotisation", "facture", "fee", "invoice"}
    if any(k in type_raw for k in motif_kw):
        if not transfert.get("motif"):
            transfert["motif"] = result["type_ordre"]
        result["type_ordre"] = None

    # Compte vs rue dans donneur_ordre
    donneur = result.get("donneur_ordre", {})
    compte = donneur.get("numero_compte_debit") or ""
    if re.search(r"[A-Za-z]{3,}", compte) and not re.match(r"^\d[\d\s]{8,}$", compte):
        if not donneur.get("rue"):
            donneur["rue"] = compte
        donneur["numero_compte_debit"] = None

    # paiement_frais_par : si numéro → c'est le compte frais
    pf = donneur.get("paiement_frais_par") or ""
    if re.match(r"^\d[\d\s]{8,}$", pf):
        if not donneur.get("numero_compte_frais"):
            donneur["numero_compte_frais"] = pf
        donneur["paiement_frais_par"] = None

    return result


def extract_ot(text: str, doc_id: str, model: str = OLLAMA_MODEL) -> dict:
    """
    Extrait les informations OT depuis le texte OCR.
    Retente jusqu'à 3 fois en cas d'échec de parsing JSON.
    """
    if not text or not text.strip():
        raise ValueError(f"[{doc_id}] Texte OCR vide.")

    prompt = _build_ot_prompt(text, doc_id)
    last_error = None

    for attempt in range(1, 4):
        try:
            raw = _call_ollama(prompt, model)
            result = _clean_llm_output(raw)
            return _postprocess_ot(result, doc_id)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_error = e
            print(f"  [WARN] Tentative {attempt}/3 échouée pour {doc_id} : {e}")

    raise ValueError(f"[{doc_id}] Échec après 3 tentatives. Dernière erreur : {last_error}")


# ---------------------------------------------------------------------------
# Extraction Company (statuts de société)
# ---------------------------------------------------------------------------

def _build_company_prompt(full_text: str, doc_id: str, doc_inserted: str) -> str:
    return f"""Tu es un expert en analyse de documents juridiques et commerciaux malgaches.
Analyse les documents suivants et extrais les informations structurées.

DOCUMENTS:
{full_text}

Extrais et retourne UNIQUEMENT un JSON valide (sans markdown, sans explication) avec cette structure exacte:

{{
    "id": "{doc_id}",
    "denomination_sociale": "dénomination sociale de la société, sinon null",
  "actionnaires": [
    {{
      "type": "personne_physique" ou "personne morale",
      "nom_prenoms": "nom complet si personne physique, sinon null",
      "part_sociale": "pourcentage ou null",
      "nationalite": "nationalité si personne physique, sinon null",
      "demeurant": "adresse si personne physique, sinon null"
    }}
  ],
  "mandataire_signataire": {{
    "nom_prenoms": "nom complet ou null",
    "piece_identite": "{doc_inserted}",
    "demeurant": "adresse ou null"
  }},
  "dirigeants": [
    {{
      "nom_prenoms": "nom complet",
      "role": "Administrateur Général / DG / PCA selon le type d'entité",
      "piece_identite": "{doc_inserted}",
      "date_naissance": "date ou null",
      "lieu_naissance": "lieu ou null",
      "nationalite": "nationalité ou null",
      "demeurant": "adresse ou null"
    }}
  ],
  "commissaire_aux_comptes": [
    {{
      "denomination": "nom du cabinet ou null",
      "nom_prenoms": "nom si personne physique ou null",
      "nationalite": "nationalité ou null",
      "demeurant": "adresse ou null"
    }}
  ]
}}

Règles importantes:
- Le mandataire signataire est généralement l'Administrateur Général principal ou le gérant
- Pour les dirigeants: inclure tous les administrateurs généraux (titulaire et adjoint)
- Pour les actionnaires: utiliser les informations de répartition du capital
- Si une information n'est pas disponible dans le document, mettre null
- Retourner UNIQUEMENT le JSON, rien d'autre"""


def extract_company(
    texts: list[dict],
    doc_id: str,
    doc_inserted: str = "",
    model: str = OLLAMA_MODEL,
) -> dict:
    """
    Extrait les informations de société depuis une liste de dicts {id, text}.
    """
    full_text = "\n\n---\n\n".join(item["text"] for item in texts)
    prompt = _build_company_prompt(full_text, doc_id, doc_inserted)

    raw = _call_ollama(prompt, model)
    result = _clean_llm_output(raw)
    return result