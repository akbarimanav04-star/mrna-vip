> **Student engineering project output - not real clinical guidance.**

# Clinical Safety Report - patient P0006

## Patient and risk score
- Patient P0006: age 76; 3 pre-existing condition(s); candidate dose 55.8 mcg (0.0558 mg); inflammatory biomarker 5.88 mg/L (not elevated by the project cut-off of 7 mg/L - the guideline itself gives no numeric cut-off); prior reaction history: None.
- Predicted adverse-reaction risk: **High** (probability 0.74)

## Decision
- Eligible for vaccine (standard administration): **NO - clinician review required**
- Recommended dose: **0.0000 mg** (0 mcg)

## Reasoning steps
1. Risk model output: High risk of an adverse reaction (probability 0.74).
2. Candidate dose 55.8 mcg = 0.0558 mg; patient age 76; 3 pre-existing condition(s); prior reaction history: None.
3. Guideline rule triggered - Age 65+ with two or more pre-existing conditions: secondary clinician review is required.
4. Guideline rule triggered - Three or more pre-existing conditions: secondary clinician review is required.
5. Decision: NOT eligible for standard administration pending secondary clinician review; no dose is recommended until then.

## Guideline excerpts cited (verbatim)
- "patients in this age band with two or more relevant pre-existing conditions should be flagged for secondary clinician review regardless of dose"
- "Three or more relevant pre-existing conditions at time of screening."
- "Patients should be monitored for at least 15 minutes following administration"

## Provenance
- Report drafted by: `offline-rule-based (no API key)` (attempts: 1, latency 0.004s)
- Retrieval embedding backend: `tfidf`
- Elevated-biomarker cut-off used: 7 mg/L (project assumption)

*Student engineering project output - not real clinical guidance.*
