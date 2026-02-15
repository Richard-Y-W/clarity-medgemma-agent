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

        pad_id = self.tokenizer.pad_token_id        # should be 0
        eos_id = self.tokenizer.eos_token_id        # should be 1
        eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")  # should be 106

        def run(text: str) -> str:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,

                    # IMPORTANT: sample a bit so it doesn't insta-stop into EOS/pad
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,

                    # IMPORTANT: stop on eos OR end_of_turn
                    eos_token_id=[eos_id, eot_id],

                    # IMPORTANT: pad token is real pad (0), but forbid generating it
                    pad_token_id=pad_id,
                    bad_words_ids=[[pad_id]],

                    # helps prevent immediate termination
                    min_new_tokens=16,
                )

            gen_ids = out[0, input_ids.shape[-1]:]

            # DEBUG: see if it's still trying to output special tokens
            print("DEBUG first new ids:", gen_ids[:12].tolist())
            print("DEBUG first new toks:", self.tokenizer.convert_ids_to_tokens(gen_ids[:12].tolist()))
            print("DEBUG raw decode:", repr(self.tokenizer.decode(gen_ids, skip_special_tokens=False)))

            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return decoded

        # 1) Try chat template
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        decoded = run(chat_text)

    # 2) Fallback: plain prompt
        if not decoded:
            decoded = run(prompt)

        return decoded








