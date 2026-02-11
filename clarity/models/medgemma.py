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
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # ===== DEBUG START =====
        print("\n===== DEBUG: TOKEN INFO =====")
        print("Input tokens:", input_ids.shape)
        print("Output tokens:", out.shape)

        print("\n===== DEBUG: FULL DECODE =====")
        full_decoded = self.tokenizer.decode(out[0], skip_special_tokens=False)
        print(full_decoded)

        print("\n===== DEBUG: NEW TOKENS ONLY =====")
        gen_ids = out[0, input_ids.shape[-1]:]
        print("New token count:", gen_ids.shape)
        print(self.tokenizer.decode(gen_ids, skip_special_tokens=False))
        print("===== END DEBUG =====\n")
        # ===== DEBUG END =====


        # ✅ decode only the newly generated tokens
        gen_ids = out[0, input_ids.shape[-1]:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return decoded.strip()


