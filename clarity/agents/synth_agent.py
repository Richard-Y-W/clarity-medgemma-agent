from clarity.schemas import PatientState, SOAPNote
from clarity.models.medgemma import MedGemmaModel


class SynthesisAgent:
    """
    Uses MedGemma to synthesize structured clinical documentation (SOAP).
    """

    def __init__(self, model: MedGemmaModel):
        self.model = model

    def generate_soap(self, state: PatientState) -> SOAPNote:
        prompt = (
            "You are a clinical documentation assistant. "
            "Generate a concise SOAP note. Do not invent facts.\n\n"
            f"Presenting complaint: {state.presenting_complaint}\n"
            f"HPI: {state.history_of_present_illness}\n"
            f"Vitals: {state.vitals}\n"
            f"Meds: {state.medications}\n"
            f"Allergies: {state.allergies}\n\n"
            "Return in this exact format:\n"
            "SUBJECTIVE: ...\nOBJECTIVE: ...\nASSESSMENT: ...\nPLAN: ...\n"
        )

        text = self.model.generate(prompt, max_new_tokens=256)

        # Very simple parsing for now (we'll harden later)
        def grab(prefix: str) -> str:
            if prefix not in text:
                return ""
            part = text.split(prefix, 1)[1]
            # stop at next field if present
            for nxt in ["SUBJECTIVE:", "OBJECTIVE:", "ASSESSMENT:", "PLAN:"]:
                if nxt != prefix and nxt in part:
                    part = part.split(nxt, 1)[0]
            return part.strip()

        return SOAPNote(
            subjective=grab("SUBJECTIVE:"),
            objective=grab("OBJECTIVE:"),
            assessment=grab("ASSESSMENT:"),
            plan=grab("PLAN:"),
        )
