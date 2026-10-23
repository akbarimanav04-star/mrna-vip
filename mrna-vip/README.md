# mRNA-VIP: Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform

Student engineering project. **Output is not real clinical guidance.**

| Module | File | What it does |
|---|---|---|
| 1 | `src/data_pipeline.py`, `src/risk_model.py` | clean 500 patient records, train Logistic Regression + Random Forest, keep the better F1 |
| 2 | `src/attention.py` | scaled dot-product self-attention over an mRNA sequence, NumPy only |
| 3 | `src/rag_index.py`, `src/rag_eval.py` | chunk the guideline text, embed, store in ChromaDB, retrieve top-k |
| 4 | `src/report_generator.py` (+ `schemas.py`, `guideline_rules.py`) | LLM report validated by Pydantic, with guardrails |
| 5 | `src/dashboard.py` | one 3-panel figure built from the saved outputs |
| 6 | `streamlit_app.py` | interactive Streamlit dashboard: overview, patient safety check, model performance, attention explorer, guideline search, data explorer |

## Run it
```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py         # runs everything, writes to outputs/
python -m pytest tests -q      # 45 tests
streamlit run streamlit_app.py # interactive dashboard (run_pipeline.py first)
```
Optional: copy `.env.example` to `.env` and add an OpenAI or Anthropic key for a real LLM, and
`pip install sentence-transformers` (internet needed once) for the real MiniLM embeddings.

## Be honest about what was run
* The reports in `outputs/reports/` were produced by the **offline rule-based generator** (no API key
  was available), and retrieval used the **TF-IDF fallback** (MiniLM could not be downloaded). The real
  LLM and MiniLM code paths are written but were not executed here. Re-run `python run_pipeline.py`
  with a key and `sentence-transformers` installed to regenerate every number and report.
* `outputs/rag_meta.json` and each report's Provenance section record which back-end was used.

## Problems in the PRD template that this code fixes
1. `pd.read_csv` turns the word **"None"** (no prior reaction) into a missing value: 291 patients would
   be imputed as "Mild". Fixed with `keep_default_na=False, na_values=[""]`.
2. The template scaled the full data before splitting (leakage) and saved only the forest. Here the scaler
   lives in a Pipeline fitted on the training split and the whole pipeline is saved.
3. `LabelEncoder` orders history alphabetically (Mild < Moderate < None). Here: None < Mild < Moderate.
4. The data is in **micrograms** but the report field is `Recommended_Dose_mg`; doses are converted (100 mcg = 0.1 mg).
5. FR-2.1 says length 5-50 but its own example `ACGU` has 4 letters; the library accepts 1-50, the pipeline requires 5+.
6. The template's chunk size (500) counted characters, FR-3.1 says tokens. Tokens are used (about 4 chars each).
7. The template's attention weights were re-drawn on every call; here they are seeded.
8. "Elevated biomarker" has no number in the guideline. The 7.0 mg/L cut-off in `config.py` is a project assumption.

## Results (from this run)
* Classifier: Logistic Regression chosen (test F1 0.70, ROC-AUC 0.77; 5-fold CV F1 0.72) vs Random Forest (F1 0.64).
* RAG: 4 chunks of 326-386 tokens; with so short a document, k=3 returns most of it (see `outputs/rag_eval.md`).

## Requirement traceability
Each FR-x.y from the PRD is covered by a test in `tests/` (named `test_fr_x_y_...`).

## What was added in the completion pass
* `notebooks/01_data_eda.ipynb` and `02_attention_math.ipynb` (executed, outputs saved).
* `REPORT.md` - the PRD Section 10 portfolio report. `docs/rag_writeup.md` - chunking choices and three failure cases.
* `docs/BEGINNER_COMPLETE_GUIDE.md` - beginner guide: every click, every file's full code, expected output of every command.
* `docs/IMPLEMENTATION_GUIDE.md` - step-by-step: which tool/app to use and what to run at each stage.
* **Attention fix:** `W_q/W_k/W_v` were scaled by 0.1, which made all scores about 0 and the heatmap almost uniform (0.064-0.069). `WEIGHT_SCALE = 0.5` now gives a visible pattern (0.01-0.15).
* Attention result: A1/A5/A9/A15 attend to each other with identical weight (0.147) because same-base positions share an embedding and there is no positional encoding.

* `streamlit_app.py` + `tests/test_module6_streamlit.py` (5 headless tests) and `docs/COMMANDS_AND_STREAMLIT_GUIDE.md`.
* `check_setup.py` - run `python check_setup.py` to find missing files/packages before running anything; `docs/FRESHER_STEP_BY_STEP.md` is now the precise v2 beginner guide.

## Fastest way to see output
Windows: double-click `run.bat` (or `.\run.bat`). Mac/Linux: `bash run.sh`. It sets up the environment, installs packages and opens the Streamlit dashboard, which builds all outputs itself on first open (sidebar button: *Rebuild all outputs*).

## Windows fix and deployment notes
* `rag_eval.py` no longer crashes on Windows (ChromaDB file lock); optional steps cannot stop `run_pipeline.py`.
* `python debug_app.py` prints what the dashboard renders (useful for bug reports). Deployment steps: `docs/DEPLOY_TO_STREAMLIT.md`.
