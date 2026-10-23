"""Central configuration: paths and project-wide constants.

Every other module imports from here so that a path or a threshold is changed in ONE place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

RAW_CSV = DATA_DIR / "patient_records.csv"
CLEAN_CSV = DATA_DIR / "patient_records_clean.csv"
GUIDELINES_TXT = DATA_DIR / "mrna_safety_guidelines.txt"

MODEL_PATH = OUTPUT_DIR / "risk_model.joblib"
CHROMA_DIR = OUTPUT_DIR / "chroma_store"
COLLECTION_NAME = "mrna_safety_guidelines"

SEED = 42

# ---- Module 1 ----
REQUIRED_COLUMNS = [
    "Patient_ID", "Age", "Biomarker_Level", "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg", "Prior_Reaction_History", "Reaction_Score",
]
NUMERIC_COLS = ["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]
CATEGORICAL_COL = "Prior_Reaction_History"
# Order matters: None < Mild < Moderate (an ORDINAL scale, not an alphabetical one).
HISTORY_ORDER = ["None", "Mild", "Moderate"]
COUNT_COL = "Pre_Existing_Conditions"
TARGET_COL = "Reaction_Score"
HIGH_RISK_PROBABILITY = 0.5   # predicted probability at/above which the label is "High"

# ---- Module 2 ----
EMBED_DIM = 8
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 50
DEFAULT_SEQUENCE = "ACGUACGUACGGCUA"

# ---- Module 3 ----
CHUNK_SIZE_TOKENS = 400       # max tokens per chunk (FR-3.1: 300-500)
CHUNK_OVERLAP_TOKENS = 50     # FR-3.1: ~50
MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 500
# Split on single newlines first. With the 300-500 token rule this gives 4 chunks for the
# ~1,300-token guideline document (so k=3 is still selective). See README "Design notes".
CHUNK_SEPARATORS = ["\n", ". ", " ", ""]
DEFAULT_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_LOG = OUTPUT_DIR / "rag_query_log.jsonl"
RAG_META = OUTPUT_DIR / "rag_meta.json"

# ---- Module 4 ----
# The guideline says "meaningfully above the normal reference range" but gives NO number.
# This is an ENGINEERING ASSUMPTION (about the 75th percentile of the provided cohort),
# not clinical guidance. Change it if your mentor specifies a cut-off.
ELEVATED_BIOMARKER_MGL = 7.0
REQUIRED_DISCLAIMER = "Student engineering project output - not real clinical guidance."

# The guideline text says doses "above 100 micrograms" need secondary review (Section 2.1).
# The code ALSO parses this number out of the retrieved chunks (guideline_rules.parse_dose_bound_mcg);
# this constant is only the fallback used to decide which patients are affected.
GUIDELINE_DOSE_REVIEW_MCG = 100.0
