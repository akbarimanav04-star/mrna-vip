> **Student engineering project output - not real clinical guidance.**

# Clinical Safety Report - patient P0021

## Patient and risk score
- Patient P0021: age 52; 1 pre-existing condition(s); candidate dose 102.7 mcg (0.1027 mg); inflammatory biomarker 9.22 mg/L (ELEVATED by the project cut-off of 7 mg/L - the guideline itself gives no numeric cut-off); prior reaction history: Mild.
- Predicted adverse-reaction risk: **High** (probability 0.87)

## Decision
- Eligible for vaccine (standard administration): **NO - clinician review required**
- Recommended dose: **0.0000 mg** (0 mcg)

## Reasoning steps
1. Risk model output: High risk of an adverse reaction (probability 0.87).
2. Candidate dose 102.7 mcg = 0.1027 mg; patient age 52; 1 pre-existing condition(s); prior reaction history: Mild.
3. Guideline rule triggered - Candidate dose is above 100 micrograms: secondary clinician review is required.
4. Guideline rule triggered - Dose above 100 micrograms combined with an elevated biomarker: secondary clinician review is required.
5. Guideline rule triggered - Elevated inflammatory biomarker at screening: secondary clinician review is required.
6. Decision: NOT eligible for standard administration pending secondary clinician review; no dose is recommended until then.

## Guideline excerpts cited (verbatim)
- "Doses above 100 micrograms are considered high-reactogenicity doses and require secondary clinician review before administration."
- "Candidate dose exceeding 100 micrograms combined with an elevated biomarker level."
- "should have dosing deferred until the biomarker returns to a normal range, or should receive the lowest clinically appropriate dose tier with enhanced post-administration monitoring"
- "Patients should be monitored for at least 15 minutes following administration"

## Provenance
- Report drafted by: `offline-rule-based (no API key)` (attempts: 1, latency 0.003s)
- Retrieval embedding backend: `tfidf`
- Elevated-biomarker cut-off used: 7 mg/L (project assumption)

*Student engineering project output - not real clinical guidance.*
