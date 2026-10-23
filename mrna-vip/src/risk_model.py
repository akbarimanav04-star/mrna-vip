"""Module 1b - train and evaluate the adverse-reaction risk classifier.

Differences from the PRD code template (each one fixes a real problem):
  1. The scaler/encoder live INSIDE a scikit-learn Pipeline and are fitted on the TRAINING
     split only. (The template scaled the whole dataset before splitting = data leakage.)
  2. Prior_Reaction_History is encoded with an explicit ORDER (None < Mild < Moderate).
     (LabelEncoder sorts alphabetically: Mild=0, Moderate=1, None=2 - the wrong order.)
  3. The WHOLE pipeline is saved to risk_model.joblib, so Week 4 can score a raw patient row
     without needing separate scaler/encoder files. (The template saved only the forest.)
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.data_pipeline import load_clean
from src.config import (
    CATEGORICAL_COL, CLEAN_CSV, COUNT_COL, FIG_DIR, HIGH_RISK_PROBABILITY, HISTORY_ORDER,
    MODEL_PATH, NUMERIC_COLS, OUTPUT_DIR, SEED, TARGET_COL,
)

FEATURE_COLS = NUMERIC_COLS + [CATEGORICAL_COL, COUNT_COL]


def build_preprocessor() -> ColumnTransformer:
    """Standardize numeric columns (FR-1.3), ordinal-encode history, pass the count through."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLS),
            ("hist", OrdinalEncoder(categories=[HISTORY_ORDER]), [CATEGORICAL_COL]),
            ("cnt", "passthrough", [COUNT_COL]),
        ]
    )


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            [("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]
        ),
        "random_forest": Pipeline(
            [("prep", build_preprocessor()),
             ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED))]
        ),
    }


def split_data(df: pd.DataFrame, seed: int = SEED):
    """80/20 split, stratified on the target, fixed seed (FR-1.4)."""
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def _save_confusion_matrix(name, y_test, preds):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels=["Low risk", "High risk"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def _save_report_figure(name, report_dict):
    rows = [[k, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1-score']:.2f}", int(v["support"])]
            for k, v in report_dict.items() if isinstance(v, dict)]
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["class", "precision", "recall", "f1", "support"], loc="center")
    tbl.scale(1, 1.4)
    ax.set_title(f"Classification report - {name} (held-out test set)")
    fig.savefig(FIG_DIR / f"classification_report_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = split_data(df)
    models = build_models()
    results, fitted = {}, {}
    scores = pd.DataFrame({"y_true": y_test.values}, index=y_test.index)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        report_dict = classification_report(y_test, preds, output_dict=True,
                                            target_names=["Low risk", "High risk"])
        report_txt = classification_report(y_test, preds, target_names=["Low risk", "High risk"])
        print(f"--- {name} ---\n{report_txt}")
        cm = confusion_matrix(y_test, preds)
        print(cm, "\n")
        (OUTPUT_DIR / f"classification_report_{name}.txt").write_text(report_txt)
        _save_confusion_matrix(name, y_test, preds)
        _save_report_figure(name, report_dict)
        results[name] = {
            "test_f1": round(f1_score(y_test, preds), 4),
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
            "test_accuracy": round(report_dict["accuracy"], 4),
            "cv5_f1_mean": round(float(cross_val_score(models[name], df[FEATURE_COLS], df[TARGET_COL],
                                                       cv=cv, scoring="f1").mean()), 4),
            "confusion_matrix": cm.tolist(),
        }
        fitted[name] = model
        scores[f"score_{name}"] = proba

    # FR-1.7: choose by test-set F1 (ties broken by ROC-AUC).
    best = max(results, key=lambda n: (results[n]["test_f1"], results[n]["test_roc_auc"]))
    scores["score_chosen"] = scores[f"score_{best}"]
    scores.to_csv(OUTPUT_DIR / "test_predictions.csv", index_label="row_index")
    summary = {"chosen_model": best, "n_train": len(X_train), "n_test": len(X_test), "models": results}
    (OUTPUT_DIR / "model_comparison.json").write_text(json.dumps(summary, indent=2))
    joblib.dump(fitted[best], MODEL_PATH)
    print(f"Chosen model (highest test F1): {best} -> saved to {MODEL_PATH}")
    return fitted, summary, (X_train, X_test, y_train, y_test)


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def score_patient(model, patient: dict) -> tuple[str, float]:
    """Score ONE raw patient record. Returns (risk_label, probability_of_high_risk)."""
    row = pd.DataFrame([patient])[FEATURE_COLS]
    proba = float(model.predict_proba(row)[0, 1])
    return ("High" if proba >= HIGH_RISK_PROBABILITY else "Low"), proba


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_and_evaluate(load_clean())
