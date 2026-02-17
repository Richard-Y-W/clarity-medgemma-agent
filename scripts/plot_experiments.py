import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame, group_cols):
    metrics = [
        "parse_success","format_valid","halluc_score","omission_rate",
        "rougeL_f1_macro","score_composite","red_flag_coverage","required_questions_coverage"
    ]
    return df.groupby(group_cols)[metrics].mean().reset_index()

def bar_plot(df, x, y, title, out_png):
    plt.figure()
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def line_plot(df, x, y, title, out_png):
    plt.figure()
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

if __name__ == "__main__":
    in_path = "runs/soap_eval_ALL.jsonl"
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True, parents=True)

    df = load_jsonl(in_path)

    # Temperature curves (if present)
    if "decoding" in df.columns:
        df["temp"] = df["decoding"].apply(lambda d: d.get("temperature") if isinstance(d, dict) else None)

        calib = summarize(df.dropna(subset=["temp"]), ["temp"]).sort_values("temp")
        if len(calib) > 0:
            line_plot(calib, "temp", "halluc_score", "Temperature vs Hallucination", out_dir/"temp_vs_halluc.png")
            line_plot(calib, "temp", "omission_rate", "Temperature vs Omission", out_dir/"temp_vs_omission.png")
            line_plot(calib, "temp", "parse_success", "Temperature vs Parse Success", out_dir/"temp_vs_parse.png")

    # Prompt variant bars
    if "prompt_variant" in df.columns:
        pv = summarize(df, ["prompt_variant"])
        bar_plot(pv, "prompt_variant", "score_composite", "Prompt Variant vs Composite", out_dir/"prompt_vs_score.png")
        bar_plot(pv, "prompt_variant", "halluc_score", "Prompt Variant vs Hallucination", out_dir/"prompt_vs_halluc.png")
        bar_plot(pv, "prompt_variant", "format_valid", "Prompt Variant vs Format Valid", out_dir/"prompt_vs_format.png")
