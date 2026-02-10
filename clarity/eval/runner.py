from clarity.eval.io import read_jsonl
from clarity.schemas import PatientState
from clarity.agents.risk_agent import RiskAgent


def main():
    path = "data/cases_eval.jsonl"
    cases = list(read_jsonl(path))
    print(f"Loaded {len(cases)} eval cases from {path}")

    agent = RiskAgent()

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

        gt = c["ground_truth"]
        gt_escalate = bool(gt["escalate"])

        print("\n---")
        print("case_id:", c["case_id"])
        print("pred escalate:", pred.escalate, "pred risk:", pred.risk_score)
        print("gt escalate:", gt_escalate)
        print("pred red_flags:", pred.red_flags)
        print("pred rationale:", pred.rationale)


if __name__ == "__main__":
    main()
