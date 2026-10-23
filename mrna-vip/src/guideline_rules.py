"""Deterministic reading of the guideline text.

An LLM drafts the report, but a safety-critical decision must not depend on the LLM being
right. This module encodes the guideline rules that apply to this dataset's columns, so that
(1) the offline report generator can run without an API key, and (2) every report - from any
LLM - is independently checked against the retrieved guideline text (guardrails).
Every rule carries the EXACT guideline sentence it comes from, and a rule can only be applied
if that sentence was really retrieved (so nothing is invented - safety constraint 2).
"""
import re
from dataclasses import dataclass
from typing import Callable

from src.config import ELEVATED_BIOMARKER_MGL, GUIDELINE_DOSE_REVIEW_MCG, HIGH_RISK_PROBABILITY


def norm(text: str) -> str:
    """Collapse whitespace and drop list bullets so text can be compared reliably."""
    text = re.sub(r"(?m)^\s*-\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Profile:
    patient_id: str
    age: int
    conditions: int
    dose_mcg: float
    biomarker: float
    prior_reaction: str
    risk_label: str
    risk_probability: float

    @property
    def biomarker_elevated(self) -> bool:
        return self.biomarker >= ELEVATED_BIOMARKER_MGL

    @property
    def dose_mg(self) -> float:
        return self.dose_mcg / 1000.0

    def summary(self) -> str:
        flag = "ELEVATED" if self.biomarker_elevated else "not elevated"
        return (
            f"Patient {self.patient_id}: age {self.age}; {self.conditions} pre-existing condition(s); "
            f"candidate dose {self.dose_mcg:g} mcg ({self.dose_mg:.4f} mg); inflammatory biomarker "
            f"{self.biomarker:.2f} mg/L ({flag} by the project cut-off of {ELEVATED_BIOMARKER_MGL:g} mg/L - "
            f"the guideline itself gives no numeric cut-off); prior reaction history: {self.prior_reaction}."
        )


@dataclass
class Rule:
    rule_id: str
    description: str
    snippet: str                       # exact guideline wording this rule is based on
    applies: Callable[[Profile], bool]


RULES = [
    Rule("DOSE_OVER_100", "Candidate dose is above 100 micrograms",
         "Doses above 100 micrograms are considered high-reactogenicity doses and require secondary clinician review before administration.",
         lambda p: p.dose_mcg > GUIDELINE_DOSE_REVIEW_MCG),
    Rule("AGE65_TWO_CONDITIONS", "Age 65+ with two or more pre-existing conditions",
         "patients in this age band with two or more relevant pre-existing conditions should be flagged for secondary clinician review regardless of dose",
         lambda p: p.age >= 65 and p.conditions >= 2),
    Rule("THREE_CONDITIONS", "Three or more pre-existing conditions",
         "Three or more relevant pre-existing conditions at time of screening.",
         lambda p: p.conditions >= 3),
    Rule("PRIOR_MODERATE", "Prior moderate reaction to an mRNA product",
         "Prior history of a moderate reaction to an mRNA-based product, even if the reaction has since resolved.",
         lambda p: p.prior_reaction == "Moderate"),
    Rule("DOSE_AND_BIOMARKER", "Dose above 100 micrograms combined with an elevated biomarker",
         "Candidate dose exceeding 100 micrograms combined with an elevated biomarker level.",
         lambda p: p.dose_mcg > GUIDELINE_DOSE_REVIEW_MCG and p.biomarker_elevated),
    Rule("BIOMARKER_ELEVATED", "Elevated inflammatory biomarker at screening",
         "should have dosing deferred until the biomarker returns to a normal range, or should receive the lowest clinically appropriate dose tier with enhanced post-administration monitoring",
         lambda p: p.biomarker_elevated),
]

MONITORING_SNIPPET = "Patients should be monitored for at least 15 minutes following administration"


def snippet_in_chunks(snippet: str, chunks: list[str]) -> bool:
    return norm(snippet).lower() in norm(" ".join(chunks)).lower()


def parse_dose_bound_mcg(chunks: list[str]):
    """Find the dose limit stated in the retrieved text, e.g. 'Doses above 100 micrograms ...'.

    Returns the number in micrograms, or None if the retrieved text does not state one.
    """
    m = re.search(r"(?:above|exceeding|over)\s+(\d+(?:\.\d+)?)\s*micrograms", " ".join(chunks), re.I)
    return float(m.group(1)) if m else None


@dataclass
class Finding:
    rule: Rule
    triggered: bool
    snippet_found: bool


def evaluate_rules(profile: Profile, chunks: list[str]) -> list[Finding]:
    return [Finding(r, r.applies(profile), snippet_in_chunks(r.snippet, chunks)) for r in RULES]


def decide(profile: Profile, chunks: list[str]) -> dict:
    """Rule-based decision used by the offline generator and by the guardrail cross-check."""
    findings = evaluate_rules(profile, chunks)
    triggered = [f for f in findings if f.triggered]
    missing = [f for f in triggered if not f.snippet_found]
    high_ml = profile.risk_label == "High" or profile.risk_probability >= HIGH_RISK_PROBABILITY
    eligible = not triggered and not high_ml and bool(chunks)
    return {"findings": findings, "triggered": triggered, "missing_support": missing,
            "high_ml_risk": high_ml, "eligible": eligible}
