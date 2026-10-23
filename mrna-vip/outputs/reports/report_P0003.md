> **Student engineering project output - not real clinical guidance.**

# Clinical Safety Report - patient P0003

## Patient and risk score
- Patient P0003: age 62; 1 pre-existing condition(s); candidate dose 124.6 mcg (0.1246 mg); inflammatory biomarker 1.92 mg/L (not elevated by the project cut-off of 7 mg/L - the guideline itself gives no numeric cut-off); prior reaction history: None.
- Predicted adverse-reaction risk: **Low** (probability 0.30)

## Decision
- Eligible for vaccine (standard administration): **NO - clinician review required**
- Recommended dose: **0.0000 mg** (0 mcg)

## Reasoning steps
1. Risk model output: Low risk of an adverse reaction (probability 0.30).
2. Candidate dose 124.6 mcg = 0.1246 mg; patient age 62; 1 pre-existing condition(s); prior reaction history: None.
3. Guideline rule triggered - Candidate dose is above 100 micrograms: secondary clinician review is required.
4. Decision: NOT eligible for standard administration pending secondary clinician review; no dose is recommended until then.

## Guideline excerpts cited (verbatim)
- "Doses above 100 micrograms are considered high-reactogenicity doses and require secondary clinician review before administration."
- "Patients should be monitored for at least 15 minutes following administration"

## Provenance
- Report drafted by: `offline-rule-based (no API key)` (attempts: 1, latency 0.003s)
- Retrieval embedding backend: `tfidf`
- Elevated-biomarker cut-off used: 7 mg/L (project assumption)

*Student engineering project output - not real clinical guidance.*
