from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel


@dataclass
class MedGemmaModel:
    """
    Thin wrapper around a HuggingFace MedGemma/Gemma instruction-tuned model.

    Key behavior:
    - Uses chat-template formatting (apply_chat_template) for best instruction-following.
    - Decodes ONLY newly-generated tokens (token-slicing), avoiding prompt-echo issues.
    """
    model_id: str
    hf_token_env: str = "HF_TOKEN"
    device: Optional[str] = None  # reserved; using device_map="auto" by default

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    model: Optional[PreTrainedModel] = None

    def load(self) -> None:
        token = os.environ.get(self.hf_token_env)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [{"role": "user", "content": prompt}]

        # Build chat text then tokenize to get attention_mask reliably
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

    # Ensure attention_mask exists (it will, but be defensive)
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_new_tokens=32,              # prevents empty generations
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

    # Decode only the newly generated tokens (no fragile string slicing)
        gen_ids = out[0, enc["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

