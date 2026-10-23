> **Student engineering project output - not real clinical guidance.**

# Clinical Safety Report - patient P0012

## Patient and risk score
- Patient P0012: age 84; 1 pre-existing condition(s); candidate dose 120.6 mcg (0.1206 mg); inflammatory biomarker 3.35 mg/L (not elevated by the project cut-off of 7 mg/L - the guideline itself gives no numeric cut-off); prior reaction history: Moderate.
- Predicted adverse-reaction risk: **High** (probability 0.79)

## Decision
- Eligible for vaccine (standard administration): **NO - clinician review required**
- Recommended dose: **0.0000 mg** (0 mcg)

## Reasoning steps
1. Risk model output: High risk of an adverse reaction (probability 0.79).
2. Candidate dose 120.6 mcg = 0.1206 mg; patient age 84; 1 pre-existing condition(s); prior reaction history: Moderate.
3. Guideline rule triggered - Candidate dose is above 100 micrograms: secondary clinician review is required.
4. Guideline rule triggered - Prior moderate reaction to an mRNA product: secondary clinician review is required.
5. Decision: NOT eligible for standard administration pending secondary clinician review; no dose is recommended until then.

## Guideline excerpts cited (verbatim)
- "Doses above 100 micrograms are considered high-reactogenicity doses and require secondary clinician review before administration."
- "Prior history of a moderate reaction to an mRNA-based product, even if the reaction has since resolved."
- "Patients should be monitored for at least 15 minutes following administration"

## Provenance
- Report drafted by: `offline-rule-based (no API key)` (attempts: 1, latency 0.003s)
- Retrieval embedding backend: `tfidf`
- Elevated-biomarker cut-off used: 7 mg/L (project assumption)

*Student engineering project output - not real clinical guidance.*
