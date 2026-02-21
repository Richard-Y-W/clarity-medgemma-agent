from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from clarity.schemas import PatientState
from clarity.models.medgemma import MedGemmaModel
from clarity.agents.synth_agent import SynthesisAgent, SoapGenConfig
from clarity.eval.soap_metrics import evaluate_soap, parse_reference_soap




@dataclass
class DecodingConfig:
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    min_new_tokens: int = 0
    max_new_tokens: int = 256


def _flatten_soap_reference(soap_ref: Dict[str, Any]) -> str:
    """Flatten dict SOAP reference into a single string for debugging/analysis."""
    if not isinstance(soap_ref, dict):
        soap_ref = {}
    subj = soap_ref.get("subjective", "")
    obj = soap_ref.get("objective", "")
    asmt = soap_ref.get("assessment", "")
    plan = soap_ref.get("plan", "")
    return (
        f"SUBJECTIVE: {subj}\n"
        f"OBJECTIVE: {obj}\n"
        f"ASSESSMENT: {asmt}\n"
        f"PLAN: {plan}\n"
    ).strip()


import re

def _normalize_pred_for_eval(raw: str) -> str:
    import re

    s = (raw or "").strip()

    # Strip markdown bold markers globally
    s = s.replace("**", "")

    # Normalize header lines (any capitalization) to exact expected tokens
    s = re.sub(r"(?im)^\s*subjective\s*:\s*", "SUBJECTIVE: ", s)
    s = re.sub(r"(?im)^\s*objective\s*:\s*", "OBJECTIVE: ", s)
    s = re.sub(r"(?im)^\s*assessment\s*:\s*", "ASSESSMENT: ", s)
    s = re.sub(r"(?im)^\s*plan\s*:\s*", "PLAN: ", s)

    # Remove fenced code blocks (models sometimes dump ```json ...```)
    s = re.sub(r"(?s)```.*?```", "", s).strip()

    # Normalize unicode dash bullets (– — −) to "- "
    s = re.sub(r"(?m)^\s*[\u2013\u2014\u2212]\s+", "- ", s)

    # Convert common bullets to "- "
    s = re.sub(r"(?m)^\s*[\*\u2022]\s+", "- ", s)
    s = re.sub(r"(?m)^\s*\d+\.\s+", "- ", s)
    m = re.search(r"(?im)^\s*SUBJECTIVE\s*:", s)
    if m:
        s = s[m.start():]
    # --- Clamp PLAN to 1–4 valid "- " bullet lines ---
    m = re.search(r"(?is)\bPLAN:\s*(.*)$", s)
    if m:
        plan = m.group(1).strip()

        lines = [ln.strip() for ln in plan.splitlines() if ln.strip()]
        norm = []

        for ln in lines:
            # Stop if model starts emitting new header
            if re.match(r"^(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN)\s*:", ln, flags=re.I):
                break

            # Normalize bullets again defensively
            ln = re.sub(r"^\s*[\*\u2022]\s+", "- ", ln)
            ln = re.sub(r"^\s*\d+\.\s+", "- ", ln)
            ln = re.sub(r"^\s*[\u2013\u2014\u2212]\s+", "- ", ln)

            # Remove junk like "- - - -"
            if re.fullmatch(r"[-\s]+", ln):
                continue

            if not ln.startswith("- "):
                ln = "- " + ln

            # Ensure content exists beyond "- "
            if len(ln.strip()) > 2:
                norm.append(ln)

        # Limit to 4 bullets
        norm = norm[:4]

        # Ensure PLAN is never empty
        if not norm:
            norm = ["- UNKNOWN"]

        # Rewrite PLAN section cleanly
        s = re.sub(
            r"(?is)\bPLAN:\s*.*$",
            "PLAN:\n" + "\n".join(norm),
            s,
        ).strip()

    return s.strip()


def run_soap_eval(
    cases_path: str,
    model_id: str,
    prompt_variant: str,
    decoding: DecodingConfig,
    out_jsonl: str,
) -> None:
    # --- helpers ---
    def soap_dict_to_string(d: Dict[str, str]) -> str:
        """Turn {'subjective':..,'objective':..,'assessment':..,'plan':..} into a canonical reference SOAP string."""
        return (
            f"SUBJECTIVE: {d.get('subjective','')}\n"
            f"OBJECTIVE: {d.get('objective','')}\n"
            f"ASSESSMENT: {d.get('assessment','')}\n"
            f"PLAN: {d.get('plan','')}\n"
        ).strip()

    # NOTE: cases file has UTF-8 BOM → use utf-8-sig
    with open(cases_path, "r", encoding="utf-8-sig") as fh:
        cases = [json.loads(line) for line in fh if line.strip()]

    model = MedGemmaModel(model_id=model_id)
    model.load()

    synth = SynthesisAgent(
        model,
        cfg=SoapGenConfig(max_new_tokens=decoding.max_new_tokens),
    )

    rows: List[Dict[str, Any]] = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in cases:
            # Always define gt (prevents NameError)
            gt = c.get("ground_truth") or {}

            state = PatientState(
                presenting_complaint=c["presenting_complaint"],
                history_of_present_illness=c.get("history_of_present_illness"),
                medications=c.get("medications", []),
                allergies=c.get("allergies", []),
                vitals=c.get("vitals"),
                age=c.get("age"),
                sex=c.get("sex"),
            )

            # Build prompt

            case_text = synth._format_case(state)

            gt = c.get("ground_truth") or {}
            rq = gt.get("required_questions", []) or []
            rf = gt.get("red_flags", []) or []

            if rq:
                case_text += "\nRequired questions (answer with value or UNKNOWN, keep labels verbatim): " + "; ".join(rq)

            if rf:
                case_text += "\nRed flags (do NOT add new ones; mention only if supported by CASE): " + "; ".join(rf)


            if hasattr(synth, "_build_prompt"):
                prompt = synth._build_prompt(case_text)
            elif hasattr(synth, "build_prompt"):
                prompt = synth.build_prompt(case_text)
            else:
                raise RuntimeError("SynthesisAgent has no prompt builder method (expected _build_prompt or build_prompt).")
            
            # Generate
            raw = model.generate(
                prompt,
                max_new_tokens=decoding.max_new_tokens,
                min_new_tokens=80,
                do_sample=decoding.do_sample,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
            )

            raw = _normalize_pred_for_eval(raw)

            # --- Reference handling ---
            # Prefer the explicit string in the cases file:
            reference_soap_str = (c.get("reference_soap") or "").strip()

            # If missing, fall back to ground_truth.soap_reference dict (if present)
            if not reference_soap_str:
                gt_ref_dict = gt.get("soap_reference") or {}
                if isinstance(gt_ref_dict, dict) and gt_ref_dict:
                    reference_soap_str = soap_dict_to_string(gt_ref_dict)

            # Parse the reference string into HEADER sections (SUBJECTIVE:/OBJECTIVE:/ASSESSMENT:/PLAN:)
            ref_sections, ref_ok = parse_reference_soap(reference_soap_str)

            # Map to evaluate_soap() expected dict keys (subjective/objective/assessment/plan)
            soap_ref = {
                "subjective": ref_sections.get("SUBJECTIVE:", ""),
                "objective": ref_sections.get("OBJECTIVE:", ""),
                "assessment": ref_sections.get("ASSESSMENT:", ""),
                "plan": ref_sections.get("PLAN:", ""),
            }

            # Hard fail if reference is unusable (prevents silent all-zeros)
            if (not ref_ok) or (not any(v.strip() for v in soap_ref.values())):
                raise RuntimeError(
                    f"[soap_runner] reference SOAP missing/unparseable for case_id={c.get('case_id')} "
                    f"ref_ok={ref_ok} ref_len={len(reference_soap_str)}"
                )
            
            print("DEBUG soap_ref lens:",
                {k: len((soap_ref.get(k,"") or "").strip()) for k in ["subjective","objective","assessment","plan"]})


            metrics = evaluate_soap(
                raw_output=raw,
                soap_ref=soap_ref,
                red_flags=gt.get("red_flags", []),
                required_questions=gt.get("required_questions", []),
                source_vitals=c.get("vitals", ""),
                source_meds=c.get("medications", []),
                source_age=c.get("age"),
                source_sex=c.get("sex"),
            )

            row: Dict[str, Any] = {
                "case_id": c.get("case_id"),
                "prompt_variant": prompt_variant,
                "model_id": model_id,
                "decoding": asdict(decoding),

                "raw_output": raw,

                # Debug outputs so your later inspection works:
                "reference_soap": reference_soap_str,
                "soap_ref": soap_ref,

                # Keep these for analysis / scoring breakdown:
                "required_questions": gt.get("required_questions", []),
                "red_flags": gt.get("red_flags", []),
                "escalate": gt.get("escalate", False),

                **metrics,
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    keys = [
        "parse_success",
        "format_valid",
        "halluc_score",
        "omission_rate",
        "rougeL_f1_macro",
        "concept_recall_macro",
        "score_composite",
    ]
    print("\n=== SOAP EVAL SUMMARY ===")
    for k in keys:
        vals = [r[k] for r in rows if (k in r and isinstance(r[k], (int, float)))]
        if vals:
            print(f"{k}: mean={sum(vals)/len(vals):.3f}  n={len(vals)}")


