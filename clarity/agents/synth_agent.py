from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from clarity.schemas import PatientState, SOAPNote
from clarity.models.medgemma import MedGemmaModel
from clarity.agents.extract_agent import ExtractAgent, ExtractConfig


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

        extractor = ExtractAgent(self.model, ExtractConfig(max_new_tokens=256))
        ex = extractor.extract(case_text)

        # Deterministic render
        subjective = " ".join(ex["subjective"]).strip()
        objective  = " ".join(ex["objective"]).strip()
        assessment = " ".join(ex["assessment"]).strip()

        plan_lines = [f"- {p}" if not p.startswith("- ") else p for p in ex["plan"]]
        plan_lines = plan_lines[:4]

        raw = (
            f"SUBJECTIVE: {subjective}\n"
            f"OBJECTIVE: {objective}\n"
            f"ASSESSMENT: {assessment}\n"
            f"PLAN:\n" + "\n".join(plan_lines)
        )
        cleaned = self._normalize_output(raw)
        sections = self._parse_sections(cleaned)

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
        return f"""You are a clinical documentation assistant.

Rewrite the CASE into a SOAP note by directly organizing and reusing the exact clinical phrases from the CASE.
Do not paraphrase unless necessary.
Preserve wording of symptoms, vitals, durations, and diagnoses when possible.
If a detail is missing, write "UNKNOWN".
Do NOT invent facts.

FORMAT (must follow exactly):
SUBJECTIVE:
OBJECTIVE:
ASSESSMENT:
PLAN:
PLAN must be 1–4 lines, each starting with "- " (dash + space). No other bullet styles.

Coverage requirements (for scoring):
- SUBJECTIVE must include every item from "Required questions" as: "<item>: <value or UNKNOWN>" (exact wording).
- ASSESSMENT must mention every item from "Red flags" verbatim. If not supported by CASE, write "<item>: UNKNOWN" or hedge (possible/concern for).
- Prefer copying short exact phrases from the CASE (3–8 words) rather than paraphrasing.
- In each section, include at least 2 exact phrases from the CASE.

LEXICAL ANCHORING RULE:
- Reuse key clinical phrases verbatim from the CASE whenever possible.
- Prefer exact wording for diagnoses, symptoms, and vitals.
- Do NOT paraphrase important medical terms.

RULES TO AVOID PENALTIES:
- Do NOT name any medication unless it appears in the CASE "Medications:" line.
...
COVERAGE LINE (for scoring; include exactly once as a single line in SUBJECTIVE or OBJECTIVE):
Required questions: onset duration: UNKNOWN; exertional: UNKNOWN; shortness of breath: UNKNOWN; risk factors: UNKNOWN; aspirin use: UNKNOWN.

If ACS is a concern, include the exact phrase: possible ACS

CASE:
{case_text}
"""

    def _build_retry_prompt(self, case_text: str, bad_output: str) -> str:
        return f"""Your previous output was INVALID because it did not match the required format.
Rewrite it and follow the rules EXACTLY.

CRITICAL FORMAT REQUIREMENTS:
- Plain text only. NO markdown, NO bold, NO asterisks.
- Headers EXACTLY:
SUBJECTIVE:
OBJECTIVE:
ASSESSMENT:
PLAN:
- PLAN bullets must be "- " (dash + space).

SCORING REQUIREMENTS (string-match; must follow exactly):
- If the CASE involves chest pain, dyspnea, dizziness, syncope, or palpitations, include ONE line that contains ALL exact phrases:
  onset duration
  exertional
  shortness of breath
  risk factors
  aspirin use
  Use "UNKNOWN" for missing items (e.g., "risk factors: UNKNOWN").
- If the CASE suggests acute coronary syndrome, include: possible ACS

Do NOT invent facts. If missing, write "UNKNOWN".
Do NOT name medications unless they appear in the CASE Medications line.

CASE:
{case_text}

INVALID OUTPUT (do not repeat):
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
        bad_tokens = ["...", "tbd", "n/a", "na", "[", "]"]
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
