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

    # MedGemma already has pad/eos; do NOT add tokens (avoids resize OOM)
        if self.tokenizer.pad_token_id is None:
            # safest fallback: reuse eos as pad (no vocab resize)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Ensure config knows pad token (no resizing)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id





  


        def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model not loaded. Call load() first.")

            def run(text: str) -> str:
                enc = self.tokenizer(text, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.model.device)
                attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,       # ✅ EOS only
                        pad_token_id=self.tokenizer.eos_token_id,       # ✅ pad with EOS to avoid <pad> spam
                    )

                gen_ids = out[0, input_ids.shape[-1]:]
                # If the model immediately ended, this can be mostly EOS/pad.
                decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                return decoded

        # 1) Try chat template (good for instruction-tuned)
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            decoded = run(chat_text)

            # 2) Fallback: plain prompt (bypasses chat formatting if model short-circuits)
            if not decoded:
                decoded = run(prompt)

            return decoded






