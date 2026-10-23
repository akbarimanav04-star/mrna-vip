"""Week-1 EDA figures (used by notebooks/01_data_eda.ipynb): class balance, distributions,
correlation heatmap, missing-value chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIG_DIR, HISTORY_ORDER, NUMERIC_COLS, TARGET_COL
from src.data_pipeline import load_clean, load_raw


def make_figures():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw, df = load_raw(), load_clean()
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(4, 4))
    counts = df[TARGET_COL].value_counts().sort_index()
    ax.bar(["Low risk (0)", "High risk (1)"], counts.values, color=["#0F6E6E", "#C0392B"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Class balance (Reaction_Score)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_class_balance.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.histplot(data=df, x=col, hue=TARGET_COL, kde=True, ax=ax, palette=["#0F6E6E", "#C0392B"])
        ax.set_title(f"{col} by risk class")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_distributions.png", dpi=150); plt.close(fig)

    enc = df.copy()
    enc["Prior_Reaction_History"] = enc["Prior_Reaction_History"].map({k: i for i, k in enumerate(HISTORY_ORDER)})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(enc.drop(columns="Patient_ID").corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (history encoded None=0, Mild=1, Moderate=2)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_correlation.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    miss = raw.isnull().sum()
    miss = miss[miss > 0]
    axes[0].bar(miss.index, miss.values, color="#E67E22")
    axes[0].set_title("Missing values in the raw file")
    axes[0].tick_params(axis="x", rotation=20)
    rate = df.groupby("Prior_Reaction_History")[TARGET_COL].mean().reindex(HISTORY_ORDER)
    axes[1].bar(rate.index, rate.values, color="#1F3A5F")
    axes[1].set_title("High-risk rate by prior reaction history"); axes[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_missing_and_history.png", dpi=150); plt.close(fig)
    print("EDA figures saved to", FIG_DIR)


if __name__ == "__main__":
    make_figures()
