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

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pad_id = self.tokenizer.pad_token_id  # usually 0
        eos_id = self.tokenizer.eos_token_id  # usually 1

        # Some tokenizers may not have <end_of_turn>; guard it.
        eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is None or eot_id < 0:
            eot_id = eos_id

        def run(text: str) -> str:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,

                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min(16, max_new_tokens),  # don't exceed max

                # sampling to avoid greedy choosing special tokens
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,

                # stop on eos OR end_of_turn
                    eos_token_id=[eos_id, eot_id],

                # IMPORTANT: if padding is needed, pad with EOS (not <pad>)
                    pad_token_id=eos_id,

                # IMPORTANT: forbid generating <pad>
                    bad_words_ids=[[pad_id]] if pad_id is not None else None,
                )

            gen_ids = out[0, input_ids.shape[-1]:]

        # DEBUG: see what's being generated
            first = gen_ids[:12].tolist()
            print("DEBUG first new ids:", first)
            print("DEBUG first new toks:", self.tokenizer.convert_ids_to_tokens(first))
            print("DEBUG raw decode:", repr(self.tokenizer.decode(gen_ids, skip_special_tokens=False)))

            return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # 1) Try chat-template prompt
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        decoded = run(chat_text)

    # 2) Fallback: raw prompt (if chat formatting causes short-circuit)
        if not decoded:
            decoded = run(prompt)

        return decoded








