"""Module 2 - scaled dot-product self-attention over an mRNA sequence, from scratch (NumPy).

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Differences from the PRD template: weights are drawn from a SEEDED random generator and kept
in a `params` dict, so the same sequence always gives the same attention map (the template
re-drew W_q/W_k/W_v from global state on every call, so results changed run to run).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_SEQUENCE, EMBED_DIM, FIG_DIR, MAX_SEQ_LEN, OUTPUT_DIR, SEED

VOCAB = ["A", "C", "G", "U"]


# Std-dev of the random projection weights. 0.1 made every score ~0, so softmax returned ~1/seq_len
# everywhere (a flat, uninformative heatmap). 0.5 gives visibly different attention per pair.
WEIGHT_SCALE = 0.5


def init_params(embed_dim: int = EMBED_DIM, d_k: int | None = None, seed: int = SEED) -> dict:
    """Toy 'learnable' parameters: one embedding vector per base + W_q, W_k, W_v."""
    d_k = d_k or embed_dim
    rng = np.random.default_rng(seed)
    return {
        "embedding": {b: rng.standard_normal(embed_dim) for b in VOCAB},
        "W_q": rng.standard_normal((embed_dim, d_k)) * WEIGHT_SCALE,
        "W_k": rng.standard_normal((embed_dim, d_k)) * WEIGHT_SCALE,
        "W_v": rng.standard_normal((embed_dim, d_k)) * WEIGHT_SCALE,
    }


PARAMS = init_params()


def validate_sequence(sequence: str, min_len: int = 1, max_len: int = MAX_SEQ_LEN) -> str:
    """Reject anything that is not a string of A/C/G/U with an allowed length (FR-2.2).

    No silent fixing: lowercase letters and DNA's 'T' are rejected, not converted.
    (FR-2.1 says length 5-50 but its own example, 'ACGU', has length 4, so the library
    function allows 1-50; the application entry point passes min_len=5 - see run().)
    """
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Sequence must be a non-empty string of A, C, G, U.")
    bad = sorted(set(sequence) - set(VOCAB))
    if bad:
        raise ValueError(f"Invalid nucleotide(s) {bad}: only A, C, G, U are allowed.")
    if not (min_len <= len(sequence) <= max_len):
        raise ValueError(f"Sequence length {len(sequence)} outside allowed range {min_len}-{max_len}.")
    return sequence


def encode_sequence(sequence: str, params: dict | None = None, min_len: int = 1) -> np.ndarray:
    """'ACGU...' -> (seq_len, embed_dim) matrix of stacked base embeddings (FR-2.1)."""
    params = params or PARAMS
    validate_sequence(sequence, min_len=min_len)
    return np.stack([params["embedding"][b] for b in sequence])


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)  # subtract the row max: numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def self_attention(X: np.ndarray, d_k: int | None = None, params: dict | None = None):
    """Single-head scaled dot-product attention.

    Returns (attention_weights (seq_len, seq_len), output (seq_len, d_k)).
    """
    params = params or PARAMS
    d_k = d_k or params["W_q"].shape[1]
    Q, K, V = X @ params["W_q"], X @ params["W_k"], X @ params["W_v"]
    scores = (Q @ K.T) / np.sqrt(d_k)           # raw similarity of every position pair
    weights = softmax(scores)                   # every row is a probability distribution
    return weights, weights @ V


def multi_head_attention(X: np.ndarray, n_heads: int = 2, seed: int = SEED):
    """BONUS: run `n_heads` independent heads and concatenate their outputs."""
    embed_dim = X.shape[1]
    all_w, outs = [], []
    for h in range(n_heads):
        p = init_params(embed_dim=embed_dim, seed=seed + 100 + h)
        w, o = self_attention(X, params=p)
        all_w.append(w)
        outs.append(o)
    return all_w, np.concatenate(outs, axis=-1)


def plot_attention(weights: np.ndarray, sequence: str, save_path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap="viridis")
    ticks = range(len(sequence))
    labels = [f"{b}{i + 1}" for i, b in enumerate(sequence)]   # base + position, e.g. A1, C2
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Attending TO position")
    ax.set_ylabel("Attending FROM position")
    ax.set_title("mRNA Sequence Self-Attention Weights")
    fig.colorbar(im, label="Attention weight")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def top_pairs(weights: np.ndarray, sequence: str, n: int = 5):
    """Strongest attention pairs, ignoring each position's attention to itself."""
    w = weights.copy()
    np.fill_diagonal(w, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-w, axis=None), w.shape))[0][:n]
    return [(f"{sequence[i]}{i + 1}", f"{sequence[j]}{j + 1}", float(weights[i, j])) for i, j in idx]


def run(sequence: str = DEFAULT_SEQUENCE):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    validate_sequence(sequence, min_len=5)
    X = encode_sequence(sequence)
    weights, _ = self_attention(X)
    assert np.allclose(weights.sum(axis=1), 1.0), "Softmax rows must sum to 1"
    plot_attention(weights, sequence, FIG_DIR / "attention_heatmap.png")
    # Saved so the dashboard (Module 5) is generated FROM this output, not hard-coded (FR-5.2).
    np.save(OUTPUT_DIR / "attention_weights.npy", weights)
    (OUTPUT_DIR / "attention_meta.json").write_text(json.dumps({
        "sequence": sequence, "embed_dim": EMBED_DIM, "seed": SEED,
        "top_pairs": top_pairs(weights, sequence)}, indent=2))
    heads, _ = multi_head_attention(X, n_heads=2)
    print(f"Sequence: {sequence}  ->  weights {weights.shape}; rows sum to 1.0")
    print("Top off-diagonal pairs:", top_pairs(weights, sequence, 3))
    print(f"Bonus multi-head: {len(heads)} heads of shape {heads[0].shape}")
    return weights


if __name__ == "__main__":
    run()
