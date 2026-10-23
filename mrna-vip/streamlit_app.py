"""Streamlit dashboard for the mRNA-VIP platform.

Run from the project root (venv active):   streamlit run streamlit_app.py

Every number on every tab is produced by the project's own modules or read from the files they
saved in outputs/. Nothing here is hard-coded. Student engineering project - NOT clinical guidance.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # lets "from src import ..." work however Streamlit is started

from src import attention as at                                   # noqa: E402
from src import report_generator as rg                            # noqa: E402
from src.config import (                                          # noqa: E402
    CLEAN_CSV, ELEVATED_BIOMARKER_MGL, FIG_DIR, HISTORY_ORDER, MODEL_PATH, OUTPUT_DIR,
    REQUIRED_DISCLAIMER, TARGET_COL,
)
from src.data_pipeline import load_clean                          # noqa: E402
from src.rag_index import build_index, load_index                 # noqa: E402
from src.risk_model import load_model                             # noqa: E402

st.set_page_config(page_title="mRNA-VIP Safety & Dosage Intelligence", page_icon="🧬", layout="wide")

NEEDED = [MODEL_PATH, CLEAN_CSV, OUTPUT_DIR / "model_comparison.json", OUTPUT_DIR / "test_predictions.csv",
          OUTPUT_DIR / "attention_weights.npy", OUTPUT_DIR / "attention_meta.json", OUTPUT_DIR / "latencies.json"]


# --------------------------------------------------------------------------- cached loaders
# st.cache_* keeps these in memory so the app does not reload the model on every click.
@st.cache_resource
def get_model():
    """Load the saved model. If the file is unreadable (for example it was saved with different
    package versions than the ones running here), rebuild everything once and load again."""
    try:
        return load_model()
    except Exception:
        import run_pipeline
        run_pipeline.main()
        return load_model()


@st.cache_resource
def get_index():
    try:
        return load_index()
    except Exception:                      # index folder missing -> build it once
        return build_index(verbose=False)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_clean()


@st.cache_data
def get_json(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text())


# --------------------------------------------------------------------------- chart helpers
def roc_figure(preds: pd.DataFrame, columns: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, col in columns.items():
        fpr, tpr, _ = roc_curve(preds["y_true"], preds[col])
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.set(title="ROC curve (held-out test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def heatmap_figure(weights: np.ndarray, sequence: str, title="Self-attention weights") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(weights, cmap="viridis")
    labels = [f"{b}{i + 1}" for i, b in enumerate(sequence)]
    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(len(sequence)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set(title=title, xlabel="Attending TO position", ylabel="Attending FROM position")
    fig.colorbar(im, ax=ax, fraction=0.046, label="weight")
    fig.tight_layout()
    return fig


def latency_figure(lat: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["RAG retrieval\n(5 queries)", "Report generation\n(LLM + guardrails)"]
    vals = [lat["rag_retrieval_s"], lat["report_generation_s"]]
    bars = ax.bar(names, vals, color=["#0F6E6E", "#1F3A5F"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}s", ha="center", va="bottom")
    ax.set(title="Pipeline stage latency", ylabel="Seconds (mean per patient)")
    fig.tight_layout()
    return fig


def show(fig: plt.Figure):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- page shell
st.title("🧬 Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform")
st.warning(f"**{REQUIRED_DISCLAIMER}** This is a learning project; never use it for real medical decisions.")

def build_everything():
    """Run the whole pipeline (same as `python run_pipeline.py`) and clear Streamlit's caches."""
    import run_pipeline                       # lives in the project root, which is on sys.path
    run_pipeline.main()
    st.cache_resource.clear()
    st.cache_data.clear()


with st.sidebar:
    # Second, independent copy of the required disclaimer: it stays visible in every state of the app.
    st.caption(f"⚠️ {REQUIRED_DISCLAIMER}")
    st.header("Pipeline")
    st.caption("Re-creates every output (data, model, attention, index, reports, figures). "
               "Use it after adding an API key to .env. Takes about 30 seconds.")
    if st.button("🔄 Rebuild all outputs"):
        with st.spinner("Running the full pipeline..."):
            try:
                build_everything()
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.stop()
        st.rerun()

missing = [p.name for p in NEEDED if not p.exists()]
if missing:
    # First run: build the outputs automatically so no terminal command is needed.
    if "build_error" not in st.session_state:
        with st.spinner("First run: building all outputs (about 30 seconds)..."):
            try:
                build_everything()
                st.session_state.pop("build_error", None)
            except Exception as exc:          # show the reason instead of looping forever
                st.session_state["build_error"] = repr(exc)
        if "build_error" not in st.session_state:
            st.rerun()
    st.error(f"Could not build the outputs automatically: {st.session_state['build_error']}")
    st.info(f"Missing files: {', '.join(missing)}. Try `python run_pipeline.py` in the terminal to see the full error.")
    if st.button("Try again"):
        st.session_state.pop("build_error", None)
        st.rerun()
    st.stop()

model, index, data = get_model(), get_index(), get_data()
comparison, lat = get_json("model_comparison.json"), get_json("latencies.json")
chosen = comparison["chosen_model"]

tabs = st.tabs(["📊 Overview", "🩺 Patient safety check", "📈 Model performance",
                "🔬 Attention explorer", "🔎 Guideline search", "🗂️ Data explorer"])

# --------------------------------------------------------------------------- 1. overview
with tabs[0]:
    st.subheader("Pipeline results at a glance")
    m = comparison["models"][chosen]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Patients", len(data))
    c2.metric("Chosen model", {"logistic_regression": "LogReg", "random_forest": "RandForest"}.get(chosen, chosen),
               help=chosen.replace("_", " ").title() + " (highest test F1)")
    c3.metric("Test F1", f"{m['test_f1']:.2f}")
    c4.metric("ROC-AUC", f"{m['test_roc_auc']:.2f}")
    c5.metric("Guideline chunks", len(index.collection.get()["ids"]))
    st.caption(f"Report drafting: **{lat['provider']}** · retrieval back-end: **{lat['retrieval_backend']}** "
               "(see README: TF-IDF / offline mode means no API key or MiniLM was available when these outputs were built).")

    preds = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    meta = get_json("attention_meta.json")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**A. Risk classifier ROC**")
        show(roc_figure(preds, {chosen: "score_chosen"}))
    with b:
        st.markdown("**B. mRNA self-attention**")
        show(heatmap_figure(np.load(OUTPUT_DIR / "attention_weights.npy"), meta["sequence"], "Self-attention"))
    with c:
        st.markdown("**C. Stage latency**")
        show(latency_figure(lat))

    reports = sorted((OUTPUT_DIR / "reports").glob("report_*.json"))
    if reports:
        st.subheader("Saved sample reports")
        rows = []
        for r in reports:
            d = json.loads(r.read_text())
            rows.append({"Patient": r.stem.replace("report_", ""), "Eligible": d["Eligible_For_Vaccine"],
                         "Dose (mg)": d["Recommended_Dose_mg"], "Citations": len(d["FDA_Safety_Warnings_Cited"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 2. patient safety check
with tabs[1]:
    st.subheader("Run the full pipeline for one patient")
    st.caption("Model 1 scores the risk → Module 3 retrieves guideline text → Module 4 drafts a validated report → "
               "guardrails re-check it against the retrieved text.")

    source = st.radio("Patient source", ["Existing patient from the dataset", "Enter a new patient"], horizontal=True)
    base = {"Patient_ID": "NEW", "Age": 50, "Biomarker_Level": 4.8, "Pre_Existing_Conditions": 1,
            "Vaccine_Dose_mcg": 85.0, "Prior_Reaction_History": "None"}
    if source.startswith("Existing"):
        pid = st.selectbox("Patient ID", data["Patient_ID"].tolist(), index=2)
        base = data[data["Patient_ID"] == pid].iloc[0].to_dict()

    with st.form("patient_form"):
        f1, f2, f3 = st.columns(3)
        age = f1.number_input("Age (years)", 18, 85, int(base["Age"]), help="Training data covers ages 18-85.")
        biomarker = f2.number_input("Inflammatory biomarker (mg/L)", 0.0, 30.0, float(base["Biomarker_Level"]), 0.1,
                                    help=f"Project assumption: elevated at ≥ {ELEVATED_BIOMARKER_MGL:g} mg/L.")
        conditions = f3.number_input("Pre-existing conditions", 0, 4, int(base["Pre_Existing_Conditions"]))
        g1, g2 = st.columns(2)
        dose = g1.number_input("Candidate dose (mcg)", 0.0, 200.0, float(base["Vaccine_Dose_mcg"]), 0.5,
                               help="Training data covers 25-150 mcg. The report converts to mg (100 mcg = 0.1 mg).")
        history = g2.selectbox("Prior reaction history", HISTORY_ORDER,
                               index=HISTORY_ORDER.index(str(base["Prior_Reaction_History"])))
        force_offline = st.checkbox("Force the offline rule-based generator (ignore any API key)", value=False)
        submitted = st.form_submit_button("Generate safety report", type="primary")

    if submitted:
        row = {"Patient_ID": base["Patient_ID"], "Age": age, "Biomarker_Level": biomarker,
               "Pre_Existing_Conditions": conditions, "Vaccine_Dose_mcg": dose, "Prior_Reaction_History": history}
        try:
            profile = rg.make_profile(row, model)
            chunks, rag_s = rg.retrieve_for_patient(index, profile)
            llm, label = rg.get_llm("offline" if force_offline else None)
            gen = rg.generate_report(profile, chunks, llm=llm, provider_label=label)
        except Exception as exc:                          # show a readable message, not a stack trace
            st.error(f"Could not generate the report: {exc}")
            st.stop()

        rep = gen.report
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### Risk model")
            st.progress(min(max(profile.risk_probability, 0.0), 1.0),
                        text=f"{profile.risk_label} risk · probability {profile.risk_probability:.2f}")
            st.markdown("#### Decision")
            if rep.Eligible_For_Vaccine:
                st.success("✅ Eligible under standard administration")
            else:
                st.error("⛔ NOT eligible – secondary clinician review required")
            st.metric("Recommended dose", f"{rep.Recommended_Dose_mg:.4f} mg",
                      help=f"= {rep.Recommended_Dose_mg * 1000:g} mcg")
            st.caption(f"Drafted by `{gen.provider}` · attempts {gen.attempts} · "
                       f"LLM stage {gen.llm_latency_s:.3f}s · retrieval {rag_s:.3f}s")
        with right:
            st.markdown("#### Reasoning steps")
            for i, step in enumerate(rep.Reasoning_Steps, 1):
                (st.warning if step.startswith("GUARDRAIL") else st.write)(f"{i}. {step}")
            st.markdown("#### Guideline excerpts cited (verbatim)")
            for cite in rep.FDA_Safety_Warnings_Cited or ["(none)"]:
                st.info(cite)
            if gen.notes:
                st.caption("Guardrails changed the draft: " + " | ".join(gen.notes))

        with st.expander("Retrieved guideline chunks sent to the report generator"):
            for i, ch in enumerate(chunks, 1):
                st.text(f"--- chunk {i} ---\n{ch}")
        with st.expander("Patient summary used in the prompt"):
            st.write(profile.summary())

        st.info(rep.Disclaimer)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download report (.json)", json.dumps(rep.model_dump(), indent=2),
                           f"report_{profile.patient_id}.json", "application/json")
        d2.download_button("⬇ Download report (.md)", rg.report_markdown(profile, gen, index.backend),
                           f"report_{profile.patient_id}.md", "text/markdown")

# --------------------------------------------------------------------------- 3. model performance
with tabs[2]:
    st.subheader("Logistic Regression vs Random Forest")
    table = pd.DataFrame(comparison["models"]).T[["test_f1", "test_roc_auc", "test_accuracy", "cv5_f1_mean"]]
    table.columns = ["Test F1", "ROC-AUC", "Accuracy", "5-fold CV F1"]
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8cf"), width="stretch")
    st.caption(f"Chosen (highest test F1): **{chosen}** · {comparison['n_train']} train rows / {comparison['n_test']} test rows. "
               "A 100-row test set is small, so treat differences of a few points with caution.")

    preds = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    left, right = st.columns(2)
    with left:
        show(roc_figure(preds, {"Logistic Regression": "score_logistic_regression",
                                "Random Forest": "score_random_forest"}))
    with right:
        st.markdown("**Confusion matrices** (rows = truth, columns = prediction)")
        for name, res in comparison["models"].items():
            cm = pd.DataFrame(res["confusion_matrix"], index=["True low", "True high"],
                              columns=["Pred low", "Pred high"])
            st.write(name.replace("_", " ").title())
            st.dataframe(cm, width="stretch")
        st.caption("'True high / Pred low' are the dangerous misses: a high-risk patient called low-risk.")

    name = st.selectbox("Classification report", list(comparison["models"]))
    st.code((OUTPUT_DIR / f"classification_report_{name}.txt").read_text())

# --------------------------------------------------------------------------- 4. attention explorer
with tabs[3]:
    st.subheader("Self-attention on an mRNA sequence")
    st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")
    seq = st.text_input("Sequence (A, C, G, U only; 5-50 letters)", at.DEFAULT_SEQUENCE).strip()
    n_heads = st.slider("Attention heads (bonus: more than 1 = multi-head)", 1, 4, 1)
    try:
        at.validate_sequence(seq, min_len=5)
    except ValueError as exc:
        st.error(str(exc))
    else:
        X = at.encode_sequence(seq)
        w, _ = at.self_attention(X)
        rows_ok = bool(np.allclose(w.sum(axis=1), 1.0))
        k1, k2, k3 = st.columns(3)
        k1.metric("Encoded shape", f"{X.shape[0]} × {X.shape[1]}")
        k2.metric("Weights shape", f"{w.shape[0]} × {w.shape[1]}")
        k3.metric("Every row sums to 1", "yes" if rows_ok else "NO")
        if n_heads == 1:
            show(heatmap_figure(w, seq))
        else:
            heads, concat = at.multi_head_attention(X, n_heads=n_heads)
            cols = st.columns(n_heads)
            for i, (col, h) in enumerate(zip(cols, heads), 1):
                with col:
                    show(heatmap_figure(h, seq, f"Head {i}"))
            st.caption(f"Concatenated output shape: {concat.shape}")
        st.markdown("**Strongest off-diagonal pairs**")
        st.dataframe(pd.DataFrame(at.top_pairs(w, seq, 8), columns=["From", "To", "Weight"]),
                     hide_index=True, width="stretch")
        st.caption("Weights are random and untrained, and there is no positional encoding, so identical bases receive "
                   "identical weights. This demonstrates the mechanism; it is not a biological finding.")

# --------------------------------------------------------------------------- 5. guideline search
with tabs[4]:
    st.subheader("Ask the guideline document")
    samples = ["What are the storage requirements below -20C?", "How long is a thawed vial stable in the refrigerator?",
               "Which doses need secondary clinician review?", "What are the absolute contraindications to the vaccine?",
               "When must a moderate or severe adverse event be documented?", "(type my own question)"]
    pick = st.selectbox("Sample questions", samples)
    question = st.text_input("Question", "" if pick == samples[-1] else pick)
    k = st.slider("Top-k chunks", 1, 4, 3)
    if st.button("Search", type="primary") and question.strip():
        found, latency = index.query(question.strip(), k)
        st.success(f"Retrieved {len(found)} chunk(s) in {latency * 1000:.1f} ms using the **{index.backend}** back-end")
        for i, ch in enumerate(found, 1):
            with st.expander(f"Result {i} (most similar first)", expanded=(i == 1)):
                st.text(ch)
        st.caption("A vector search always returns its nearest chunks, even if none answers the question. "
                   "Always read the result; the report generator additionally verifies every citation.")
    rag_md = OUTPUT_DIR / "rag_eval.md"
    if rag_md.exists():
        with st.expander("Retrieval evaluation (chunk-size experiment)"):
            st.markdown(rag_md.read_text())

# --------------------------------------------------------------------------- 6. data explorer
with tabs[5]:
    st.subheader("Cleaned patient dataset")
    f1, f2, f3 = st.columns(3)
    risk = f1.multiselect("Reaction score", [0, 1], default=[0, 1], format_func=lambda x: "High (1)" if x else "Low (0)")
    hist = f2.multiselect("Prior reaction", HISTORY_ORDER, default=HISTORY_ORDER)
    ages = f3.slider("Age range", 18, 85, (18, 85))
    view = data[data[TARGET_COL].isin(risk) & data["Prior_Reaction_History"].isin(hist)
                & data["Age"].between(*ages)]
    st.caption(f"{len(view)} of {len(data)} patients match")
    st.dataframe(view, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        col = st.selectbox("Distribution of", ["Age", "Biomarker_Level", "Vaccine_Dose_mcg"])
        counts, edges = np.histogram(view[col], bins=20) if len(view) else (np.array([]), np.array([]))
        if len(view):
            st.bar_chart(pd.DataFrame({"patients": counts}, index=np.round(edges[:-1], 1)))
    with right:
        by = st.selectbox("Reaction rate by", ["Pre_Existing_Conditions", "Prior_Reaction_History"])
        st.bar_chart(data.groupby(by)[TARGET_COL].mean().rename("share with high reaction"))
    figs = sorted(FIG_DIR.glob("eda_*.png"))
    if figs:
        with st.expander("EDA figures from Week 1"):
            for f in figs:
                st.image(str(f))
