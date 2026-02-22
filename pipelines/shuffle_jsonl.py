#!/usr/bin/env python3
import argparse, json, random
from collections import defaultdict
from pathlib import Path

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(p, items):
    with open(p, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def interleave(groups, rng):
    # groups: dict subject -> list of items (already shuffled within subject)
    subjects = list(groups.keys())
    rng.shuffle(subjects)  # randomize which subject starts
    idx = {s: 0 for s in subjects}
    out = []
    remaining = True
    while remaining:
        remaining = False
        for s in subjects:
            i = idx[s]
            if i < len(groups[s]):
                out.append(groups[s][i])
                idx[s] += 1
                remaining = True
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--key", default="subject_id", help="field name for subject")
    ap.add_argument("--mode", choices=["balanced", "global"], default="balanced")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = list(read_jsonl(args.in_jsonl))

    if args.mode == "global":
        rng.shuffle(items)
        write_jsonl(args.out_jsonl, items)
        return

    groups = defaultdict(list)
    missing_key = 0
    for x in items:
        if args.key not in x:
            missing_key += 1
            # put missing-key items into their own bucket so they still appear
            groups["__missing__"].append(x)
        else:
            groups[str(x[args.key])].append(x)

    # shuffle inside each subject
    for s in groups:
        rng.shuffle(groups[s])

    out = interleave(groups, rng)
    write_jsonl(args.out_jsonl, out)

    if missing_key:
        print(f"Warning: {missing_key} items missing key '{args.key}'")

if __name__ == "__main__":
    main()
