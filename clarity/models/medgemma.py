from typing import Optional


class MedGemmaModel:
    """
    Thin wrapper around a HuggingFace/HAI-DEF MedGemma model.

    We keep this isolated so:
      - Kaggle notebook can load the model there
      - local dev can use a smaller model or CPU fallback
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._loaded = False

    def load(self):
        """
        Real loading will be added after we confirm the Kaggle model access path.
        """
        self._loaded = True

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not self._loaded:
            raise RuntimeError("MedGemmaModel not loaded. Call load() first.")
        # Placeholder so pipeline runs before model access is finalized
        return "[MEDGEMMA_PLACEHOLDER_OUTPUT]\n" + prompt[:200]
