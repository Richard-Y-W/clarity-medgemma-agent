# clarity/models/medgemma.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

@dataclass
class MedGemmaModel:
    model_id: str = "google/medgemma-1.5-4b-it"
    hf_token_env: str = "HF_TOKEN"

    tokenizer: Any = None
    model: Any = None

    def load(self):
        token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)

        use_4bit = os.getenv("USE_4BIT", "0") == "1"

        model_kwargs = dict(
            token=token,
            torch_dtype="auto",
            device_map="auto",
        )

        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as e:
                print(f"[WARN] USE_4BIT=1 but bitsandbytes not available; falling back. Reason: {e}")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

        # Make sure model config knows pad token (important for attention masks)
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    # ---- prompt helpers ----
    def _build_prompt(self, prompt: str) -> str:
        """
        Prefer chat template. Fallback to raw prompt if template unavailable.
        """
        try:
            # MedGemma IT expects chat format + generation prompt
            chat_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return chat_text
        except Exception:
            return prompt

    def _eos_ids(self) -> Union[int, List[int]]:
        eos_id = self.tokenizer.eos_token_id
        ids: List[int] = [eos_id] if eos_id is not None else []
        try:
            eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if eot is not None and eot >= 0 and eot not in ids:
                ids.append(eot)
        except Exception:
            pass
        if len(ids) == 1:
            return ids[0]
        return ids

    # ---- generation ----
    class _MinNewTokens(LogitsProcessor):
        def __init__(self, min_new: int, eos_ids: Sequence[int], start_len: int):
            self.min_new = int(min_new)
            self.eos_ids = set(int(x) for x in eos_ids)
            self.start_len = int(start_len)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            # block EOS until we have generated min_new tokens
            cur_len = input_ids.shape[-1]
            if (cur_len - self.start_len) < self.min_new:
                for eid in self.eos_ids:
                    if 0 <= eid < scores.shape[-1]:
                        scores[..., eid] = -float("inf")
            return scores

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 0,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        debug: bool = False,
        **gen_kwargs,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        text = self._build_prompt(prompt)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

        pad_id = self.tokenizer.pad_token_id
        eos_ids = self._eos_ids()
        eos_list = eos_ids if isinstance(eos_ids, list) else [eos_ids]

        prompt_len = input_ids.shape[-1]
        max_length = prompt_len + int(max_new_tokens)

    # Prevent PAD token generation explicitly
        bad_words = [[pad_id]] if pad_id is not None else None

    # Min-new-tokens: block EOS until enough tokens generated
        lp = LogitsProcessorList()
        if min_new_tokens and min_new_tokens > 0:
            lp.append(self._MinNewTokens(min_new_tokens, eos_list, start_len=prompt_len))

    # Repetition controls (defaults, caller can override via gen_kwargs)
        repetition_penalty = float(gen_kwargs.pop("repetition_penalty", 1.15))
        no_repeat_ngram_size = int(gen_kwargs.pop("no_repeat_ngram_size", 4))

    # Deterministic search settings (caller can override via gen_kwargs too if desired)
        num_beams = int(gen_kwargs.pop("num_beams", 4 if not do_sample else 1))
        early_stopping = bool(gen_kwargs.pop("early_stopping", (True if not do_sample else False)))

    # Only pass sampling params when sampling
        gen_temperature = float(temperature) if do_sample else None
        gen_top_p = float(top_p) if do_sample else None

        if debug:
            gc = getattr(self.model, "generation_config", None)
            print("== DEBUG MedGemma.generate ==")
            print("model_id:", self.model_id)
            print("device:", next(self.model.parameters()).device, "dtype:", next(self.model.parameters()).dtype)
            print("pad_id:", pad_id, "eos_ids:", eos_list)
            print("prompt_len:", prompt_len, "max_length:", max_length)
            print("gen_config.max_length:", getattr(gc, "max_length", None))
            print("do_sample:", do_sample, "temperature:", gen_temperature, "top_p:", gen_top_p)
            print("num_beams:", num_beams, "early_stopping:", early_stopping)
            print("repetition_penalty:", repetition_penalty, "no_repeat_ngram_size:", no_repeat_ngram_size)

            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            finite = torch.isfinite(out.logits).all().item()
            nan_count = torch.isnan(out.logits).sum().item()
            print("forward logits finite?", finite, "nan_count", nan_count)

    # Build kwargs once to avoid accidental duplicates
        hf_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,  # hard clamp
            do_sample=do_sample,
            eos_token_id=eos_list,
            pad_token_id=(self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else pad_id),
            bad_words_ids=bad_words,
            logits_processor=lp if len(lp) else None,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )

    # Add sampling params only when relevant
        if do_sample:
            hf_kwargs["temperature"] = gen_temperature
            hf_kwargs["top_p"] = gen_top_p

    # Let any remaining caller kwargs pass through (after pops above)
        hf_kwargs.update({k: v for k, v in gen_kwargs.items() if v is not None})

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            out_ids = self.model.generate(**hf_kwargs)

        new_ids = out_ids[0, prompt_len:]
        text_out = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text_out












