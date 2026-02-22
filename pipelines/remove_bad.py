import json
src="data/datasets/tb_v1/manifest_train1.jsonl"
dst="data/preprocess/tb_v1/v1/preprocess_manifest_bad_data.jsonl"

kept=0
dropped=0
with open(src) as f, open(dst,"w") as g:
    for i,line in enumerate(f):
        j=json.loads(line)
        q=(j.get("question") or "").strip()
        a=(j.get("answer_gt") or j.get("answer") or "").strip()
        if (not q) or (not a):
            dropped += 1
            continue
        g.write(json.dumps(j, ensure_ascii=False) + "\n")
        kept += 1

print("kept:", kept)
print("dropped:", dropped)
print("wrote:", dst)