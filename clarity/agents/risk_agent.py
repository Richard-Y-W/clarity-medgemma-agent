from clarity.schemas import PatientState, RiskEstimate


class RiskAgent:
    """
    Computes the Clinical Risk Surface (CRS) and escalation decision.
    """

    def estimate_risk(self, state: PatientState) -> RiskEstimate:
        """
        Estimate risk of critical omission and determine whether escalation is required.
        """
        raise NotImplementedError
