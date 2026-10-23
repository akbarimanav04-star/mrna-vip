"""Module 4 output schema (FR-4.1). Exactly five fields; anything else is rejected."""
from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import REQUIRED_DISCLAIMER


class ClinicalSafetyReport(BaseModel):
    # extra="forbid": unexpected keys fail validation.  strict=True: no silent type coercion
    # (the string "yes" is NOT accepted as a bool, "0.1" is NOT accepted as a float).
    model_config = ConfigDict(extra="forbid", strict=True)

    Eligible_For_Vaccine: bool = Field(
        ..., description="True = may proceed under standard administration; False = defer / secondary clinician review"
    )
    Recommended_Dose_mg: float = Field(
        ..., ge=0, description="Recommended dose in MILLIGRAMS (dataset is in micrograms: 100 mcg = 0.1 mg). 0.0 when not eligible."
    )
    Reasoning_Steps: List[str] = Field(
        ..., min_length=1, description="Ordered, human-readable reasoning behind the decision"
    )
    FDA_Safety_Warnings_Cited: List[str] = Field(
        ..., description="Exact guideline sentences (copied verbatim from the retrieved excerpts) the decision relies on"
    )
    Disclaimer: str = Field(default=REQUIRED_DISCLAIMER, description="Always present in the final report")

    @model_validator(mode="after")
    def _disclaimer_always_present(self):
        """FR-4.5: the required notice can never be missing or replaced by something else."""
        if REQUIRED_DISCLAIMER not in self.Disclaimer:
            self.Disclaimer = (self.Disclaimer.strip() + " " + REQUIRED_DISCLAIMER).strip()
        return self
