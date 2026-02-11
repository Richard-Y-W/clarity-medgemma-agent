from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from clarity.schemas import PatientState, SOAPNote
from clarity.models.medgemma import MedGemmaModel


@dataclass(frozen=True)
class SoapGenConfig:
    """
    Synthesis configuration tuned for instruction-tuned MedGemma.

    Goals:
    - Force non-placeholder content (ban "..." and other filler).
    - Produce a strictly parseable 4-section SOAP.
    - Keep notes concise + clinically cautious.
    """
    max_new_tokens: int = 256
    max_sentences_per_section: int = 4
    temperature: float | None = None  # not used if do_sample=False in model wrapper
    disallow_placeholders: bool = True


class SynthesisAgent:
    """
    Uses MedGemma to synthesize structured clinical documentation (SOAP).

    This agent:
    1) Builds a strict prompt that enforces format and bans placeholders.
    2) Calls MedGemma.
    3) Parses the output robustly into SUBJECTIVE/OBJECTIVE/ASSESSMENT/PLAN.
    """

    # Canonical headers
    _HEADERS: List[str] = ["SUBJECTIVE:", "OBJECTIVE:", "ASSESSMENT:", "PLAN:"]

    def __init__(self, model: MedGemmaModel, cfg: SoapGenConfig | None = None):
        self.model = model
        self.cfg = cfg or SoapGenConfig()

    def generate_soap(self, state: PatientState) -> SOAPNote:
        case_text = self._format_case(state)
        prompt = self._build_prompt(case_text)

        raw = self.model.generate(prompt, max_new_tokens=self.cfg.max_new_tokens)

        # Normalize + parse
        cleaned = self._normalize_output(raw)
        sections = self._parse_sections(cleaned)

        # Guardrail: if model returned placeholders, try one strict retry
        if self.cfg.disallow_placeholders and self._looks_like_placeholder(sections):
            retry_prompt = self._build_retry_prompt(case_text, cleaned)
            raw2 = self.model.generate(retry_prompt, max_new_tokens=self.cfg.max_new_tokens)
            cleaned2 = self._normalize_output(raw2)
            sections2 = self._parse_sections(cleaned2)
            # Use the better one (fewer empty / placeholder-like sections)
            if self._quality_score(sections2) >= self._quality_score(sections):
                sections = sections2

        return SOAPNote(
            subjective=sections.get("SUBJECTIVE:", "").strip(),
            objective=sections.get("OBJECTIVE:", "").strip(),
            assessment=sections.get("ASSESSMENT:", "").strip(),
            plan=sections.get("PLAN:", "").strip(),
        )

    # ----------------------------
    # Prompting
    # ----------------------------
    def _build_prompt(self, case_text: str) -> str:
        # NOTE: We intentionally do NOT include "SUBJECTIVE: ..." placeholders.
        # We tell the model to output headers only, with real content.
        return f"""You are a clinical documentation assistant.
Write a SOAP note ONLY using the information provided below.
If something is missing, write "UNKNOWN" (do NOT invent details).

CASE:
Presenting complaint: {state.presenting_complaint}
HPI: {state.history_of_present_illness}
Vitals: {state.vitals}
Meds: {state.medications}
Allergies: {state.allergies}

OUTPUT FORMAT (exactly):
SUBJECTIVE: <1-3 sentences>
OBJECTIVE: <1-3 sentences>
ASSESSMENT: <1-3 sentences>
PLAN: <1-4 bullet points prefixed with "- ">

Rules:
- Do NOT use "..." or placeholders.
- Do NOT invent facts not in the case. If something is missing, state what you need.
- Keep each section to at most {self.cfg.max_sentences_per_section} sentences.
- No meta commentary, no disclaimers.

CASE:
{case_text}
"""

    def _build_retry_prompt(self, case_text: str, bad_output: str) -> str:
        # A stricter retry that explicitly calls out what went wrong.
        return f"""You previously returned an invalid SOAP note (it contained placeholders like "..." or missing content).
Fix it.

CASE:
Presenting complaint: {state.presenting_complaint}
HPI: {state.history_of_present_illness}
Vitals: {state.vitals}
Meds: {state.medications}
Allergies: {state.allergies}

OUTPUT FORMAT (exactly):
SUBJECTIVE: <1-3 sentences>
OBJECTIVE: <1-3 sentences>
ASSESSMENT: <1-3 sentences>
PLAN: <1-4 bullet points prefixed with "- ">

Hard rules:
- No placeholders (no "...", no "TBD", no "N/A").
- Use only information from the case. If uncertain, write "UNKNOWN" (do not ask questions).
- Each section <= {self.cfg.max_sentences_per_section} sentences.

CASE:
{case_text}

INVALID OUTPUT (for reference; do not repeat):
{bad_output}
"""

    def _format_case(self, state: PatientState) -> str:
        # Robustly stringify fields even if the schema contains lists/dicts.
        def fmt(x) -> str:
            if x is None:
                return ""
            if isinstance(x, (list, tuple)):
                return ", ".join(str(v) for v in x)
            if isinstance(x, dict):
                return "; ".join(f"{k}: {v}" for k, v in x.items())
            return str(x)

        lines = [
            f"Presenting complaint: {fmt(getattr(state, 'presenting_complaint', ''))}",
            f"HPI: {fmt(getattr(state, 'history_of_present_illness', ''))}",
            f"Vitals: {fmt(getattr(state, 'vitals', ''))}",
            f"Medications: {fmt(getattr(state, 'medications', ''))}",
            f"Allergies: {fmt(getattr(state, 'allergies', ''))}",
        ]

        # Optional fields if present in your schema
        for optional in ["red_flags", "pregnancy_status", "age", "sex", "ground_truth"]:
            if hasattr(state, optional):
                val = fmt(getattr(state, optional))
                if val:
                    lines.append(f"{optional.replace('_', ' ').title()}: {val}")

        return "\n".join(lines).strip()

    # ----------------------------
    # Parsing & cleanup
    # ----------------------------
    def _normalize_output(self, text: str) -> str:
        # Normalize whitespace and ensure headers appear at line starts.
        t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

        # Some models echo the prompt; if they include multiple "CASE:" blocks,
        # keep the last chunk after the final "SUBJECTIVE:" occurrence.
        if "SUBJECTIVE:" in t:
            t = t.split("SUBJECTIVE:", 1)[0] + "SUBJECTIVE:" + t.split("SUBJECTIVE:", 1)[1]
        return t

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """
        Parse by locating each header and slicing until the next header.
        Works even if content spans multiple lines.
        """
        if not text:
            return {h: "" for h in self._HEADERS}

        # Ensure each header is at the start of a line (best-effort).
        for h in self._HEADERS:
            text = re.sub(rf"\s*{re.escape(h)}\s*", f"\n{h} ", text)

        sections: Dict[str, str] = {h: "" for h in self._HEADERS}

        # Find header positions
        positions = []
        for h in self._HEADERS:
            m = re.search(rf"(?m)^\s*{re.escape(h)}", text)
            if m:
                positions.append((m.start(), h))

        if not positions:
            return sections

        positions.sort(key=lambda x: x[0])
        for i, (start, h) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            chunk = text[start:end].strip()

            # Remove the header from the chunk
            chunk = re.sub(rf"(?m)^\s*{re.escape(h)}\s*", "", chunk).strip()

            # Clean trailing accidental headers
            for other in self._HEADERS:
                if other != h and other in chunk:
                    chunk = chunk.split(other, 1)[0].strip()

            sections[h] = chunk

        return sections

    # ----------------------------
    # Quality heuristics
    # ----------------------------
    def _looks_like_placeholder(self, sections: Dict[str, str]) -> bool:
        # If any section is empty or contains placeholder tokens, mark as bad.
        bad_tokens = ["...", "tbd", "n/a", "na", "unknown", "[", "]"]
        for h in self._HEADERS:
            s = (sections.get(h) or "").strip()
            if not s:
                return True
            low = s.lower()
            if any(tok in low for tok in bad_tokens):
                return True
        return False

    def _quality_score(self, sections: Dict[str, str]) -> int:
        # Higher is better: count non-empty, non-placeholder sections.
        score = 0
        for h in self._HEADERS:
            s = (sections.get(h) or "").strip()
            if s and "..." not in s and "tbd" not in s.lower():
                score += 1
        return score
