from clarity.schemas import PatientState


class IntakeAgent:
    """
    Responsible for adaptive clinical questioning.
    Determines what information is missing and asks targeted follow-up questions.
    """

    def next_question(self, state: PatientState) -> str:
        """
        Given the current patient state, return the next best question to ask.
        """
        raise NotImplementedError
