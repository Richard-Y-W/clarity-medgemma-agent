import argparse

from clarity.eval.soap_runner import run_soap_eval, DecodingConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="data/cases_eval.jsonl")
    p.add_argument("--out", default="runs/soap_eval.jsonl")
    p.add_argument("--model_id", default="google/medgemma-1.5-4b-it")
    p.add_argument("--prompt_variant", default="strict_v1")
    p.add_argument("--do_sample", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--min_new_tokens", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    run_soap_eval(
        cases_path=args.cases,
        model_id=args.model_id,
        prompt_variant=args.prompt_variant,
        decoding=DecodingConfig(
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
        ),
        out_jsonl=args.out,
    )

if __name__ == "__main__":
    main()
