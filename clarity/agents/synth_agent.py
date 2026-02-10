from clarity.schemas import PatientState, SOAPNote


class SynthesisAgent:
    """
    Uses MedGemma to synthesize structured clinical documentation.
    """

    def generate_soap(self, state: PatientState) -> SOAPNote:
        """
        Generate a structured SOAP note from the patient state.
        """
        raise NotImplementedError
