from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class MedGemmaModel:
    model_id: str
    hf_token_env: str = "HF_TOKEN"
    device: Optional[str] = None  # reserved; using device_map="auto"

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    model: Optional[PreTrainedModel] = None

    def load(self) -> None:
        token = os.environ.get(self.hf_token_env)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)

        # DO NOT add tokens / DO NOT resize embeddings.
        # If pad missing, reuse eos as pad WITHOUT changing vocab size.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise RuntimeError("Tokenizer has no pad_token_id and no eos_token; cannot set padding safely.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        # Ensure config has correct ids
        self.model.config.pad_token_id = int(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            self.model.config.eos_token_id = int(self.tokenizer.eos_token_id)

        # Defensive: avoid tiny shipped defaults like max_length=20
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.pad_token_id = int(self.tokenizer.pad_token_id)
            if self.tokenizer.eos_token_id is not None:
                self.model.generation_config.eos_token_id = int(self.tokenizer.eos_token_id)
            if getattr(self.model.generation_config, "max_length", None) is not None:
                if int(self.model.generation_config.max_length) < 256:
                    self.model.generation_config.max_length = 2048

    def _encode(self, text: str) -> dict:
        assert self.tokenizer is not None
        # CRITICAL: don't let tokenizer auto-append EOS (can cause immediate stop)
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

    def _build_chat_text(self, prompt: str) -> Optional[str]:
        assert self.tokenizer is not None
        try:
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(chat_text, str) and chat_text.strip():
                return chat_text
        except Exception:
            pass
        return None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_new = int(max(1, max_new_tokens))
        min_new = int(max(0, min_new_tokens))
        if min_new > max_new:
            min_new = 0

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if eos_id is None:
            raise RuntimeError("Tokenizer eos_token_id is None; cannot stop reliably.")

        # Pad with EOS so early-stop doesn't look like <pad>-spam
        pad_for_generation = int(eos_id)

        # Prevent generating pad as a real token when pad != eos
        bad_words = None
        if pad_id is not None and int(pad_id) != int(eos_id):
            bad_words = [[int(pad_id)]]

        def run(text: str) -> str:
            enc = self._encode(text)
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)
            prompt_len = int(input_ids.shape[-1])

            min_len = prompt_len + min_new
            max_len = prompt_len + max_new  # absolute max_length defeats tiny generation_config.max_length

            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_length=min_len,
                    max_length=max_len,
                    eos_token_id=int(eos_id),
                    pad_token_id=pad_for_generation,
                    bad_words_ids=bad_words,
                    do_sample=bool(do_sample),
                    temperature=float(temperature) if do_sample else None,
                    top_p=float(top_p) if do_sample else None,
                )

            gen_ids = out[0, prompt_len:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        chat_text = self._build_chat_text(prompt)
        if chat_text is not None:
            ans = run(chat_text)
            if ans:
                return ans

        return run(prompt)








