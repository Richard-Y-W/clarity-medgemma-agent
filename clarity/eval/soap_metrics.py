from __future__ import annotations

import re
import string
from typing import Dict, List, Tuple, Optional

HEADERS = ["SUBJECTIVE:", "OBJECTIVE:", "ASSESSMENT:", "PLAN:"]

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at",
    "with","without","as","by","from","is","are","was","were","be","been","being","this","that","these","those",
    "patient","pt","reports","report","denies","no","yes"
}

_PLACEHOLDER_RE = re.compile(r"(\.\.\.|tbd|n/?a)", re.IGNORECASE)

VITAL_PATTERNS = [
    (re.compile(r"\bbp\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", re.IGNORECASE), "BP"),
    (re.compile(r"\bhr\s*[:=]?\s*(\d{2,3})\b", re.IGNORECASE), "HR"),
    (re.compile(r"\brr\s*[:=]?\s*(\d{1,2})\b", re.IGNORECASE), "RR"),
    (re.compile(r"\bspo2\s*[:=]?\s*(\d{2,3})\s*%?\b", re.IGNORECASE), "SpO2"),
    (re.compile(r"\btemp(?:erature)?\s*[:=]?\s*(\d{2}(?:\.\d)?)\s*(c|f)?\b", re.IGNORECASE), "Temp"),
]

AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:yo|y/o|year old|years old)\b", re.IGNORECASE)
SEX_RE = re.compile(r"\b(male|female|man|woman|m\b|f\b)\b", re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(possible|likely|suspect|concern for|cannot rule out)\b", re.IGNORECASE)

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("\u00a0", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s

def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split() if t]

def lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def rouge_l_f1(pred: str, ref: str) -> float:
    p = _tokens(pred)
    r = _tokens(ref)
    if not p or not r:
        return 0.0
    l = lcs_len(p, r)
    prec = l / len(p)
    rec = l / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def token_f1(pred: str, ref: str) -> float:
    p = set(_tokens(pred))
    r = set(_tokens(ref))
    if not p or not r:
        return 0.0
    tp = len(p & r)
    prec = tp / len(p)
    rec = tp / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def concept_recall(pred: str, ref: str) -> float:
    rtoks = [t for t in _tokens(ref) if len(t) >= 4 and t not in STOPWORDS]
    if not rtoks:
        return 0.0
    ptoks = set(_tokens(pred))
    hit = sum(1 for t in rtoks if t in ptoks)
    return hit / len(rtoks)

def parse_headers(raw: str) -> Tuple[Dict[str, str], bool]:
    raw = raw or ""
    idx = {h: raw.find(h) for h in HEADERS}
    if any(v == -1 for v in idx.values()):
        return {}, False
    order = [idx[h] for h in HEADERS]
    if order != sorted(order):
        return {}, False

    sections: Dict[str, str] = {}
    for i, h in enumerate(HEADERS):
        start = idx[h] + len(h)
        end = idx[HEADERS[i + 1]] if i + 1 < len(HEADERS) else len(raw)
        sections[h] = raw[start:end].strip()
    return sections, True

def plan_bullets_ok(plan: str) -> bool:
    lines = [l.strip() for l in (plan or "").splitlines() if l.strip()]
    if not lines:
        return False
    return all(l.startswith("- ") for l in lines)

def count_string_coverage(text: str, items: List[str]) -> float:
    if not items:
        return 0.0
    nt = _norm(text)
    hit = 0
    for it in items:
        if _norm(it) in nt:
            hit += 1
    return hit / len(items)

def extract_vitals(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for pat, key in VITAL_PATTERNS:
        ms = pat.findall(text or "")
        if not ms:
            continue
        vals = []
        for m in ms:
            if isinstance(m, tuple):
                vals.append("/".join([x for x in m if x]))
            else:
                vals.append(str(m))
        out[key] = vals
    return out

def extract_age(text: str) -> Optional[int]:
    m = AGE_RE.search(text or "")
    return int(m.group(1)) if m else None

def extract_sex(text: str) -> Optional[str]:
    m = SEX_RE.search(text or "")
    if not m:
        return None
    s = m.group(1).lower()
    if s in {"m", "man", "male"}:
        return "M"
    if s in {"f", "woman", "female"}:
        return "F"
    return None

def meds_from_source(source_meds: List[str]) -> set:
    return set(_norm(m) for m in (source_meds or []) if m)

def extract_med_like_phrases(text: str) -> List[str]:
    toks = _tokens(text)
    suffixes = ("pril","sartan","olol","statin","azole","caine","mab","nib","vir","cillin","mycin")
    candidates = []
    for t in toks:
        if len(t) < 5:
            continue
        if t.endswith(suffixes):
            candidates.append(t)
    for fixed in ["aspirin","tylenol","acetaminophen","ibuprofen","nitroglycerin","heparin","insulin","metformin"]:
        if fixed in toks:
            candidates.append(fixed)
    return sorted(set(candidates))

def hallucination_metrics(
    soap_text: str,
    source_vitals: str,
    source_meds: List[str],
    source_age: Optional[int],
    source_sex: Optional[str],
    ref_assessment: str,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    metrics["has_placeholder"] = 1.0 if _PLACEHOLDER_RE.search(soap_text or "") else 0.0

    # vitals hallucination
    gen_v = extract_vitals(soap_text)
    src_norm = _norm(source_vitals or "")
    extracted = 0
    halluc = 0
    for k, vals in gen_v.items():
        for v in vals:
            extracted += 1
            if _norm(v) not in src_norm and _norm(k) not in src_norm:
                halluc += 1
    metrics["vitals_claims"] = float(extracted)
    metrics["vitals_hallucinations"] = float(halluc)
    metrics["vitals_halluc_rate"] = (halluc / extracted) if extracted else 0.0

    # meds hallucination (conservative)
    src_m = meds_from_source(source_meds)
    gen_m = extract_med_like_phrases(soap_text)
    extracted = len(gen_m)
    halluc = 0
    for m in gen_m:
        if m not in src_m and "unknown" not in _norm(soap_text):
            halluc += 1
    metrics["med_claims"] = float(extracted)
    metrics["med_hallucinations"] = float(halluc)
    metrics["med_halluc_rate"] = (halluc / extracted) if extracted else 0.0

    # demographics hallucination
    gen_age = extract_age(soap_text)
    gen_sex = extract_sex(soap_text)

    demo_h = 0
    demo_c = 0
    if gen_age is not None:
        demo_c += 1
        if source_age is None:
            if "unknown" not in _norm(soap_text):
                demo_h += 1
        else:
            if gen_age != source_age:
                demo_h += 1

    if gen_sex is not None:
        demo_c += 1
        if source_sex is None:
            if "unknown" not in _norm(soap_text):
                demo_h += 1
        else:
            if gen_sex != source_sex.upper():
                demo_h += 1

    metrics["demo_claims"] = float(demo_c)
    metrics["demo_hallucinations"] = float(demo_h)
    metrics["demo_halluc_rate"] = (demo_h / demo_c) if demo_c else 0.0

    # dx hallucination flag (anchored to reference; conservative)
    ref_set = set(_tokens(ref_assessment or ""))
    gen_diag = [t for t in _tokens(soap_text) if len(t) >= 6 and t not in STOPWORDS]
    extra = [t for t in gen_diag if t not in ref_set]
    if extra and not HEDGE_RE.search(soap_text or ""):
        metrics["dx_hallucination_flag"] = 1.0
    else:
        metrics["dx_hallucination_flag"] = 0.0

    metrics["halluc_score"] = (
        0.4 * metrics["vitals_halluc_rate"]
        + 0.4 * metrics["med_halluc_rate"]
        + 0.2 * metrics["demo_halluc_rate"]
    )
    return metrics

def evaluate_soap(
    raw_output: str,
    soap_ref: Dict[str, str],
    red_flags: List[str],
    required_questions: List[str],
    source_vitals: str,
    source_meds: List[str],
    source_age: Optional[int],
    source_sex: Optional[str],
) -> Dict[str, float]:
    sections, ok = parse_headers(raw_output or "")
    parse_success = 1.0 if ok and plan_bullets_ok(sections.get("PLAN:", "")) else 0.0
    has_ph = 1.0 if _PLACEHOLDER_RE.search(raw_output or "") else 0.0
    format_valid = 1.0 if (parse_success == 1.0 and has_ph == 0.0) else 0.0

    out: Dict[str, float] = {
        "parse_success": parse_success,
        "format_valid": format_valid,
        "has_placeholder": has_ph,
        "section_nonempty_rate": 0.0,
    }

    rouge_scores = []
    f1_scores = []
    concept_scores = []
    nonempty = 0

    for h in HEADERS:
        key = h[:-1].lower()  # subjective/objective/assessment/plan
        pred = sections.get(h, "")
        ref = (soap_ref or {}).get(key, "")
        if pred.strip():
            nonempty += 1

        rl = rouge_l_f1(pred, ref)
        tf1 = token_f1(pred, ref)
        cr = concept_recall(pred, ref)
        out[f"rougeL_f1_{key}"] = rl
        out[f"token_f1_{key}"] = tf1
        out[f"concept_recall_{key}"] = cr
        rouge_scores.append(rl)
        f1_scores.append(tf1)
        concept_scores.append(cr)

    out["section_nonempty_rate"] = nonempty / 4.0
    out["rougeL_f1_macro"] = sum(rouge_scores) / 4.0
    out["token_f1_macro"] = sum(f1_scores) / 4.0
    out["concept_recall_macro"] = sum(concept_scores) / 4.0
    out["omission_rate"] = 1.0 - out["concept_recall_macro"]

    full_text = raw_output or ""
    out["red_flag_coverage"] = count_string_coverage(full_text, red_flags or [])
    out["required_questions_coverage"] = count_string_coverage(full_text, required_questions or [])

    hm = hallucination_metrics(
        soap_text=full_text,
        source_vitals=source_vitals,
        source_meds=source_meds,
        source_age=source_age,
        source_sex=source_sex,
        ref_assessment=(soap_ref or {}).get("assessment", ""),
    )
    out.update(hm)

    out["score_composite"] = (
        0.35 * out["rougeL_f1_macro"]
        + 0.25 * out["red_flag_coverage"]
        + 0.15 * out["required_questions_coverage"]
        + 0.15 * out["concept_recall_macro"]
        + 0.10 * out["format_valid"]
        - 0.35 * out["halluc_score"]
    )
    return out
