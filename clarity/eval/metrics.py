from typing import List, Dict
from clarity.schemas import RiskEstimate


def evaluate_escalation(
    predictions: List[RiskEstimate],
    ground_truth: List[Dict],
) -> Dict[str, float]:
    """
    Compute simple escalation metrics.
    """

    assert len(predictions) == len(ground_truth)

    tp = fp = tn = fn = 0

    for pred, gt in zip(predictions, ground_truth):
        gt_escalate = bool(gt["escalate"])
        if pred.escalate and gt_escalate:
            tp += 1
        elif pred.escalate and not gt_escalate:
            fp += 1
        elif not pred.escalate and not gt_escalate:
            tn += 1
        elif not pred.escalate and gt_escalate:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
    }
