from clarity.eval.io import read_jsonl
from clarity.schemas import PatientState
from clarity.agents.risk_agent import RiskAgent
from clarity.eval.metrics import evaluate_escalation


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


if __name__ == "__main__":
    main()

