from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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

    def generate(self, prompt: str, max_new_tokens: int = 256, min_new_tokens: int = 0) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        stop_ids = [i for i in [eos_id, eot_id] if i is not None and i != -1]

        def run(text: str) -> str:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

            prompt_len = input_ids.shape[-1]

            # Hard safety: never request more tokens than physically possible
            min_new = max(0, int(min_new_tokens))
            max_new = max(1, int(max_new_tokens))
            if min_new > max_new:
                min_new = 0

            # IMPORTANT: define absolute lengths to avoid generation_config.max_length=20 ruining you
            min_len = prompt_len + min_new
            max_len = prompt_len + max_new

            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,

                    min_length=min_len,
                    max_length=max_len,

                # sampling prevents "immediate stop into special token" behavior
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,

                    eos_token_id=stop_ids if stop_ids else eos_id,

                # pad with EOS (prevents <pad> spam in output tokens)
                    pad_token_id=eos_id,

                # forbid generating <pad> if it exists
                    bad_words_ids=[[pad_id]] if pad_id is not None else None,
                )

            gen_ids = out[0, prompt_len:]
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return decoded

    # 1) chat-template
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        decoded = run(chat_text)

    # 2) fallback: plain prompt
        if not decoded:
            decoded = run(prompt)

        return decoded









