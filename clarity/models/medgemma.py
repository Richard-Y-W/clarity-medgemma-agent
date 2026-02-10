from typing import Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MedGemmaModel:
    def __init__(self, model_id: str, hf_token_env: str = "HF_TOKEN", device: Optional[str] = None):
        self.model_id = model_id
        self.hf_token_env = hf_token_env
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self):
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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
