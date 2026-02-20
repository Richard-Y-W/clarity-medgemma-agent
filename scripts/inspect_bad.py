import json

path = "runs/soap_eval_greedy.jsonl"
bad = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("format_valid", 0) != 1.0:
            bad.append(r)

print("Bad:", len(bad))
for r in bad:
    print("\nCASE", r.get("case_id"), "format_valid=", r.get("format_valid"), "has_placeholder=", r.get("has_placeholder"))
    print((r.get("raw_output") or "")[:2000])
