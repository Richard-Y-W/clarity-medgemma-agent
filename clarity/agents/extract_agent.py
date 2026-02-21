# clarity/agents/extract_agent.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from clarity.models.medgemma import MedGemmaModel


@dataclass(frozen=True)
class ExtractConfig:
    max_new_tokens: int = 256


_JSON_RE = re.compile(r"(?s)\{.*\}")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_supported(span: str, case_text: str) -> bool:
    if not span:
        return False
    if span.strip().upper() == "UNKNOWN":
        return True
    return _norm(span) in _norm(case_text)


def _clean_list(xs: Any) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, str):
        xs = [xs]
    if not isinstance(xs, list):
        return []
    out = []
    for x in xs:
        if not isinstance(x, str):
            continue
        t = x.strip()
        if not t:
            continue
        out.append(t)
    return out


class ExtractAgent:
    def __init__(self, model: MedGemmaModel, cfg: ExtractConfig | None = None):
        self.model = model
        self.cfg = cfg or ExtractConfig()

    def build_prompt(self, case_text: str) -> str:
        return f"""You are performing clinical information extraction.

Return ONLY valid JSON (no markdown, no commentary).
Rules:
- Every string MUST be copied verbatim from the CASE, or be exactly "UNKNOWN".
- Keep each list 1–3 items (PLAN 1–4 items).
- Do not invent anything.

JSON schema:
{{
  "subjective": ["..."],
  "objective": ["..."],
  "assessment": ["..."],
  "plan": ["..."]
}}

CASE:
{case_text}
"""

    def extract(self, case_text: str) -> Dict[str, List[str]]:
        prompt = self.build_prompt(case_text)
        raw = self.model.generate(prompt, max_new_tokens=self.cfg.max_new_tokens)

        data = _extract_json(raw) or {}

        subj = _clean_list(data.get("subjective"))
        obj = _clean_list(data.get("objective"))
        asmt = _clean_list(data.get("assessment"))
        plan = _clean_list(data.get("plan"))

        # Validate: drop any span not supported by CASE (except UNKNOWN)
        subj = [s for s in subj if _is_supported(s, case_text)] or ["UNKNOWN"]
        obj  = [s for s in obj  if _is_supported(s, case_text)] or ["UNKNOWN"]
        asmt = [s for s in asmt if _is_supported(s, case_text)] or ["UNKNOWN"]

        # PLAN: clamp to 4, drop unsupported, ensure nonempty
        plan = [p for p in plan if _is_supported(p, case_text)]
        plan = plan[:4]
        if not plan:
            plan = ["UNKNOWN"]

        return {"subjective": subj, "objective": obj, "assessment": asmt, "plan": plan}