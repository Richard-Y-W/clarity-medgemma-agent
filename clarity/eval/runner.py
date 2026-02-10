from clarity.eval.io import read_jsonl
from clarity.schemas import PatientState
from clarity.agents.risk_agent import RiskAgent
from clarity.eval.metrics import evaluate_escalation
from clarity.models.medgemma import MedGemmaModel
from clarity.agents.synth_agent import SynthesisAgent



def main():
    path = "data/cases_eval.jsonl"
    cases = list(read_jsonl(path))
    print(f"Loaded {len(cases)} eval cases from {path}")

    agent = RiskAgent()

    predictions = []
    ground_truth = []

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
        pred = agent.estimate_risk(state)
        predictions.append(pred)
        ground_truth.append(c["ground_truth"])

    metrics = evaluate_escalation(predictions, ground_truth)

    print("\n=== Baseline RiskAgent metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

        # MedGemma (placeholder) synthesis demo on first case
    model = MedGemmaModel(model_id="MEDGEMMA_TBD")
    model.load()
    synth = SynthesisAgent(model)

    demo = cases[0]
    demo_state = PatientState(
        presenting_complaint=demo["presenting_complaint"],
        history_of_present_illness=demo.get("history_of_present_illness"),
        medications=demo.get("medications", []),
        allergies=demo.get("allergies", []),
        vitals=demo.get("vitals"),
        age=demo.get("age"),
        sex=demo.get("sex"),
    )
    soap = synth.generate_soap(demo_state)

    print("\n=== MedGemma Synthesis (demo) ===")
    print("case_id:", demo["case_id"])
    print("SUBJECTIVE:", soap.subjective[:200])
    print("OBJECTIVE:", soap.objective[:200])
    print("ASSESSMENT:", soap.assessment[:200])
    print("PLAN:", soap.plan[:200])



if __name__ == "__main__":
    main()

