from __future__ import annotations
import json, math
import pandas as pd
import matplotlib.pyplot as plt

def load(path: str):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for l in f:
            if l.strip():
                rows.append(json.loads(l))
    return pd.DataFrame(rows)

def main(path="runs/soap_eval_extract_250.jsonl", outdir="runs"):
    df = load(path)
    print(df[["score_composite","halluc_score","concept_recall_macro","rougeL_f1_macro","omission_rate","format_valid"]].describe())

    # 1) Hallucination distribution
    plt.figure()
    df["halluc_score"].hist(bins=30)
    plt.title("Hallucination Score Distribution")
    plt.xlabel("halluc_score"); plt.ylabel("count")
    plt.savefig(f"{outdir}/halluc_hist.png", dpi=200)

    # 2) Recall distribution
    plt.figure()
    df["concept_recall_macro"].hist(bins=30)
    plt.title("Concept Recall Distribution")
    plt.xlabel("concept_recall_macro"); plt.ylabel("count")
    plt.savefig(f"{outdir}/recall_hist.png", dpi=200)

    # 3) Tradeoff scatter
    plt.figure()
    plt.scatter(df["halluc_score"], df["concept_recall_macro"], s=10)
    plt.title("Recall vs Hallucination Tradeoff")
    plt.xlabel("halluc_score"); plt.ylabel("concept_recall_macro")
    plt.savefig(f"{outdir}/tradeoff.png", dpi=200)

    # 4) Composite distribution
    plt.figure()
    df["score_composite"].hist(bins=30)
    plt.title("Composite Score Distribution")
    plt.xlabel("score_composite"); plt.ylabel("count")
    plt.savefig(f"{outdir}/score_hist.png", dpi=200)

    print("saved plots to", outdir)

if __name__ == "__main__":
    main()