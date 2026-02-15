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
        if self.tokenizer.pad_token_id is None:
        # many gemma tokenizers already have "<pad>" but sometimes it isn't set
            if "<pad>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<pad>"
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id  # 0

        # If we added tokens, resize embeddings
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))


  


    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc["attention_mask"].to(self.model.device)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=32,                 # prevents immediate termination
            do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<end_of_turn>")],
            pad_token_id=self.tokenizer.pad_token_id,
        )

        gen_ids = out[0, input_ids.shape[-1]:]  # ✅ only new tokens
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return decoded





