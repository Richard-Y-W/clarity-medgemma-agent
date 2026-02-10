class SafetyAgent:
    """
    Enforces safety constraints such as refusal, uncertainty signaling,
    and non-diagnostic behavior.
    """

    def check(self, text: str) -> str:
        """
        Inspect generated text and apply safety filters or warnings if needed.
        """
        raise NotImplementedError
