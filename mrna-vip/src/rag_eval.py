"""Module 3 evaluation: sample-question table + chunk-size ablation.

A question counts as a HIT when the retrieved chunk contains the phrase that actually answers it
(`GOLD`). hit@1 = answer is in the top chunk; hit@3 = answer is somewhere in the top 3.
The ablation rebuilds the index with other chunk sizes (in a temp folder) to show the trade-off
between precision and the amount of text handed to the LLM.
"""
import gc
import json
import tempfile

from src.config import OUTPUT_DIR
from src.rag_index import build_index, count_tokens

QUESTIONS = [
    ("What are the storage requirements below -20C?", "ultra-low temperature"),
    ("How long is a thawed vial stable in the refrigerator?", "stable for up to 30 days"),
    ("Which doses need secondary clinician review?", "Doses above 100 micrograms"),
    ("How should patients aged 65 or older with two conditions be handled?", "Ages 65 and older"),
    ("What are the absolute contraindications to the vaccine?", "anaphylaxis"),
    ("When must a moderate or severe adverse event be documented?", "within 24 hours"),
    ("How long should a patient be monitored after administration?", "at least 15 minutes"),
    ("What should happen if the inflammatory biomarker is elevated?", "biomarker returns to a normal range"),
]

PROFILES = {
    "spec (400 tok / 50 overlap)": dict(chunk_size=400, overlap=50),
    "small (120 tok / 20 overlap)": dict(chunk_size=120, overlap=20),
    "tiny (60 tok / 10 overlap)": dict(chunk_size=60, overlap=10),
}


def evaluate_profile(index, k: int = 3) -> dict:
    rows, hit1, hit3, ctx_tokens, lat = [], 0, 0, [], []
    for q, gold in QUESTIONS:
        chunks, latency = index.query(q, k, log=False)
        h1, h3 = gold.lower() in chunks[0].lower(), any(gold.lower() in c.lower() for c in chunks)
        hit1, hit3 = hit1 + h1, hit3 + h3
        ctx_tokens.append(sum(count_tokens(c) for c in chunks))
        lat.append(latency)
        rows.append({"question": q, "gold_phrase": gold, "hit_at_1": bool(h1), "hit_at_3": bool(h3),
                     "latency_s": round(latency, 5), "top_chunk_preview": chunks[0][:90].replace("\n", " ")})
    n = len(QUESTIONS)
    return {"hit_at_1": hit1 / n, "hit_at_3": hit3 / n, "mean_context_tokens": sum(ctx_tokens) / n,
            "mean_latency_s": sum(lat) / n, "rows": rows}


def run():
    out = {}
    for name, cfg in PROFILES.items():
        # ignore_cleanup_errors: on Windows ChromaDB keeps its files open, so deleting the temp folder
        # raises PermissionError (WinError 32). The leftover temp folder is harmless.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            idx = build_index(persist_dir=tmp, collection_name="ablation", verbose=False, **cfg)
            res = evaluate_profile(idx)
            res["n_chunks"] = idx.collection.count()
            res["backend"] = idx.backend
            out[name] = res
            del idx
            gc.collect()                      # drop Chroma's file handles before the folder is removed
    (OUTPUT_DIR / "rag_eval.json").write_text(json.dumps(out, indent=2))
    lines = ["| Profile | Chunks | hit@1 | hit@3 | Mean tokens sent to LLM | Mean latency (s) |", "|---|---|---|---|---|---|"]
    for name, r in out.items():
        lines.append(f"| {name} | {r['n_chunks']} | {r['hit_at_1']:.0%} | {r['hit_at_3']:.0%} | "
                     f"{r['mean_context_tokens']:.0f} | {r['mean_latency_s']:.4f} |")
    first = out[next(iter(out))]
    lines += ["", "| Question | Top chunk starts with | Hit@1 | Latency (s) |", "|---|---|---|---|"]
    for r in first["rows"]:
        lines.append(f"| {r['question']} | {r['top_chunk_preview']}... | {'yes' if r['hit_at_1'] else 'NO'} | {r['latency_s']:.4f} |")
    (OUTPUT_DIR / "rag_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    run()
