"""Module 5 - one 3-panel figure built ONLY from the saved outputs of Modules 1-4 (FR-5.1..5.3).

Panel A: ROC curve  <- outputs/test_predictions.csv   (written by risk_model.py)
Panel B: attention  <- outputs/attention_weights.npy   (written by attention.py)
Panel C: latency    <- outputs/latencies.json          (written by report_generator.py)
Re-run any module, then this script, and the matching panel changes.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.config import FIG_DIR, OUTPUT_DIR


def build_dashboard(save_path=FIG_DIR / "dashboard.png"):
    preds = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    summary = json.loads((OUTPUT_DIR / "model_comparison.json").read_text())
    weights = np.load(OUTPUT_DIR / "attention_weights.npy")
    meta = json.loads((OUTPUT_DIR / "attention_meta.json").read_text())
    lat = json.loads((OUTPUT_DIR / "latencies.json").read_text())
    sequence = meta["sequence"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A - ROC curve of the chosen model
    fpr, tpr, _ = roc_curve(preds["y_true"], preds["score_chosen"])
    axes[0].plot(fpr, tpr, lw=2, label=f"{summary['chosen_model']} (AUC = {auc(fpr, tpr):.2f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    axes[0].set(title="Risk Classifier ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[0].legend(loc="lower right")

    # Panel B - attention heatmap
    im = axes[1].imshow(weights, cmap="viridis")
    labels = [f"{b}{i + 1}" for i, b in enumerate(sequence)]
    axes[1].set_xticks(range(len(sequence)))
    axes[1].set_yticks(range(len(sequence)))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set(title="mRNA Self-Attention Weights", xlabel="Attending TO position", ylabel="Attending FROM position")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="weight")

    # Panel C - latency comparison
    names = ["RAG retrieval\n(5 queries)", "Report generation\n(LLM + guardrails)"]
    vals = [lat["rag_retrieval_s"], lat["report_generation_s"]]
    bars = axes[2].bar(names, vals, color=["#0F6E6E", "#1F3A5F"])
    axes[2].set(title="Pipeline Stage Latency (seconds)", ylabel="Seconds (mean per patient)")
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}s", ha="center", va="bottom")
    axes[2].set_xlabel(f"LLM: {lat['provider']} | retrieval: {lat['retrieval_backend']}", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")
    return save_path


if __name__ == "__main__":
    build_dashboard()
