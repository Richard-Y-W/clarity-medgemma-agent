from clarity.schemas import PatientState, RiskEstimate


class RiskAgent:
    """
    Rule-based baseline Clinical Risk Surface (CRS).

    Outputs:
      - risk_score in [0,1]
      - red_flags detected (interpretable)
      - escalate decision (thresholded risk + hard red flags)
      - short rationale string (for auditability)
    """

    # Simple red-flag keyword rules.
    # (We will expand and refine, but start here.)
    RED_FLAG_RULES = {
        "possible ACS": ["chest pain", "chest pressure", "radiat", "diaphoresis", "sweating", "left arm"],
        "possible PE": ["shortness of breath", "pleuritic", "o2", "hypox", "tachypnea", "long flight", "oral contracept"],
        "possible stroke": ["slurred speech", "facial droop", "arm weakness", "difficulty walking", "sudden dizziness"],
        "possible DKA": ["type 1", "missed insulin", "fruity breath", "vomiting", "tachypnea"],
        "thunderclap headache": ["worst headache", "thunderclap", "sudden onset"],
        "appendicitis concern": ["right lower quadrant", "rlq", "migration", "rebound", "guarding"],
        "anaphylaxis concern": ["wheezing", "lip swelling", "tongue swelling", "trouble breathing"],
        "sepsis concern": ["fever", "hypotension", "confusion"],
        "dehydration risk (peds)": ["fewer wet diapers", "decreased intake", "letharg"],
    }

    # Vital-sign pattern flags (very lightweight heuristics)
    def _vital_flags(self, vitals_text: str) -> list[str]:
        if not vitals_text:
            return []
        t = vitals_text.lower()
        flags = []
        # crude parsing: just look for substrings
        if "spo2" in t:
            # flag hypoxia if contains "92" "91" "90" etc
            for bad in [" 90", " 91", " 92", " 89", " 88", " 87"]:
                if bad in t:
                    flags.append("hypoxia")
                    break
        # tachycardia heuristics
        for bad in ["hr 110", "hr 112", "hr 120", "hr 122", "hr 130", "hr 140"]:
            if bad in t:
                flags.append("tachycardia")
                break
        # fever heuristics
        for bad in ["temp 38", "temp 39", "39.5", "40"]:
            if bad in t:
                flags.append("fever")
                break
        # very high BP heuristic
        for bad in ["bp 170", "bp 180", "bp 190"]:
            if bad in t:
                flags.append("severe hypertension")
                break
        return flags

    def estimate_risk(self, state: PatientState) -> RiskEstimate:
        text = " ".join(
            [
                state.presenting_complaint or "",
                state.history_of_present_illness or "",
                state.vitals or "",
                " ".join(state.medications or []),
                " ".join(state.allergies or []),
            ]
        ).lower()

        detected = []
        # keyword flags
        for label, keys in self.RED_FLAG_RULES.items():
            for k in keys:
                if k in text:
                    detected.append(label)
                    break

        # vitals flags (add as red flags)
        vflags = self._vital_flags(state.vitals or "")
        detected.extend(vflags)

        # de-duplicate, keep stable order
        seen = set()
        red_flags = []
        for x in detected:
            if x not in seen:
                red_flags.append(x)
                seen.add(x)

        # Risk score: simple monotone function of number of red flags
        # (Interpretability > cleverness for baseline.)
        n = len(red_flags)
        # map n -> risk: 0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92...
        if n == 0:
            risk = 0.10
        elif n == 1:
            risk = 0.25
        elif n == 2:
            risk = 0.45
        elif n == 3:
            risk = 0.65
        elif n == 4:
            risk = 0.78
        else:
            risk = 0.88

        # Escalate if high risk OR hard-critical conditions
        hard_critical = any(
            rf in red_flags
            for rf in [
                "possible ACS",
                "possible PE",
                "possible stroke",
                "possible DKA",
                "thunderclap headache",
                "anaphylaxis concern",
            ]
        )
        escalate = hard_critical or (risk >= 0.60)

        rationale_parts = []
        if red_flags:
            rationale_parts.append("Detected red flags: " + ", ".join(red_flags))
        else:
            rationale_parts.append("No red flags detected from available text.")
        if escalate:
            rationale_parts.append("Escalation recommended due to elevated risk.")
        else:
            rationale_parts.append("No immediate escalation recommended based on baseline rules.")

        return RiskEstimate(
            risk_score=risk,
            rationale=" ".join(rationale_parts),
            red_flags=red_flags,
            escalate=escalate,
        )
