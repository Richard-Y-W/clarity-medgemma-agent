from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from clarity.eval.io import read_jsonl
from clarity.eval.soap_metrics import evaluate_soap
from clarity.schemas import PatientState
from clarity.models.medgemma import MedGemmaModel
from clarity.agents.synth_agent import SynthesisAgent, SoapGenConfig

@dataclass
class DecodingConfig:
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.95
    min_new_tokens: int = 0
    max_new_tokens: int = 256

def run_soap_eval(
    cases_path: str,
    model_id: str,
    prompt_variant: str,
    decoding: DecodingConfig,
    out_jsonl: str,
) -> None:
    cases = list(read_jsonl(cases_path))

    model = MedGemmaModel(model_id=model_id)
    model.load()

    # NOTE: We keep SoapGenConfig minimal here; we'll add decoding knobs to it next step.
    synth = SynthesisAgent(model, cfg=SoapGenConfig(max_new_tokens=decoding.max_new_tokens))

    rows: List[Dict[str, Any]] = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in cases:
            state = PatientState(
                presenting_complaint=c["presenting_complaint"],
                history_of_present_illness=c.get("history_of_present_illness"),
                medications=c.get("medications", []),
                allergies=c.get("allergies", []),
                vitals=c.get("vitals"),
                age=c.get("age"),
                sex=c.get("sex"),
            )

            # Build prompt using your existing SynthesisAgent internal methods.
            case_text = synth._format_case(state)
            prompt = synth._build_prompt(case_text)

            raw = model.generate(
                prompt,
                max_new_tokens=decoding.max_new_tokens,
                min_new_tokens=decoding.min_new_tokens,
                do_sample=decoding.do_sample,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
            )

            gt = c["ground_truth"]
            metrics = evaluate_soap(
                raw_output=raw,
                soap_ref=gt.get("soap_reference", {}),
                red_flags=gt.get("red_flags", []),
                required_questions=gt.get("required_questions", []),
                source_vitals=c.get("vitals", ""),
                source_meds=c.get("medications", []),
                source_age=c.get("age"),
                source_sex=c.get("sex"),
            )

            row = {
                "case_id": c["case_id"],
                "prompt_variant": prompt_variant,
                "model_id": model_id,
                "decoding": asdict(decoding),
                "raw_output": raw,
                **metrics,
            }
            f.write(json.dumps(row) + "\n")
            rows.append(row)

    keys = ["parse_success","format_valid","halluc_score","omission_rate","rougeL_f1_macro","score_composite"]
    print("\n=== SOAP EVAL SUMMARY ===")
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            print(f"{k}: mean={sum(vals)/len(vals):.3f}  n={len(vals)}")
