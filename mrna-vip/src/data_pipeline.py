"""Module 1a - load, validate, profile and clean patient_records.csv.

IMPORTANT PARSING DETAIL
------------------------
`Prior_Reaction_History` contains the literal word "None" (= the patient had no prior
reaction). pandas treats the string "None" as a missing value by default, which would make
291 healthy-history patients look "missing" and then get re-labelled "Mild" by mode
imputation. We therefore read the file with `keep_default_na=False, na_values=[""]` so only
EMPTY cells count as missing.
"""
import pandas as pd

from src.config import (
    CATEGORICAL_COL, CLEAN_CSV, HISTORY_ORDER, NUMERIC_COLS, RAW_CSV, REQUIRED_COLUMNS,
)


class MissingColumnsError(ValueError):
    """Raised when the input CSV lacks one or more required columns (FR-1.1)."""

    def __init__(self, missing):
        self.missing = sorted(missing)
        super().__init__(f"Required column(s) missing from CSV: {', '.join(self.missing)}")


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Load the CSV and validate that all 7 required columns exist (FR-1.1)."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise MissingColumnsError(missing)
    return df


def load_clean(path=CLEAN_CSV) -> pd.DataFrame:
    """Load the cleaned CSV. Same NA rules as load_raw so the category "None" survives."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def profile(df: pd.DataFrame) -> dict:
    """Shape, dtypes, missing counts and summary statistics."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe(include="all").T,
    }


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Impute Biomarker_Level with the median and Prior_Reaction_History with the mode (FR-1.2).

    Why impute instead of dropping rows? Only ~4% / ~3.4% of rows are affected, and with just
    500 patients every row is valuable. Median is robust to the right-skew of the biomarker;
    mode is the only sensible central value for a category.
    Note: medians/modes are computed on all 500 rows (as the PRD specifies a standalone
    cleaning step). With <4% missing the leakage is negligible; scaling, which matters more,
    is fitted on the training split only (see risk_model.py).
    """
    df = df.copy()
    if verbose:
        print("Missing values before cleaning:\n", df.isnull().sum(), "\n")
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())
    df[CATEGORICAL_COL] = df[CATEGORICAL_COL].fillna(df[CATEGORICAL_COL].mode()[0])
    bad = set(df[CATEGORICAL_COL].unique()) - set(HISTORY_ORDER)
    if bad:
        raise ValueError(f"Unexpected Prior_Reaction_History values: {sorted(bad)}")
    return df


def run() -> pd.DataFrame:
    df = load_raw()
    print("Shape:", df.shape)
    cleaned = clean(df)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)
    print("Missing values after cleaning:", int(cleaned.isnull().sum().sum()))
    print(f"Saved {CLEAN_CSV}")
    return cleaned


if __name__ == "__main__":
    run()
