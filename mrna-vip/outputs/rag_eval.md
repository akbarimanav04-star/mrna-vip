| Profile | Chunks | hit@1 | hit@3 | Mean tokens sent to LLM | Mean latency (s) |
|---|---|---|---|---|---|
| spec (400 tok / 50 overlap) | 4 | 100% | 100% | 1059 | 0.0023 |
| small (120 tok / 20 overlap) | 16 | 88% | 100% | 270 | 0.0025 |
| tiny (60 tok / 10 overlap) | 38 | 62% | 88% | 103 | 0.0026 |

| Question | Top chunk starts with | Hit@1 | Latency (s) |
|---|---|---|---|
| What are the storage requirements below -20C? | Section 1: Storage and Handling Requirements  1.1 Ultra-Low Temperature Storage Unopened m... | yes | 0.0050 |
| How long is a thawed vial stable in the refrigerator? | Section 1: Storage and Handling Requirements  1.1 Ultra-Low Temperature Storage Unopened m... | yes | 0.0022 |
| Which doses need secondary clinician review? | 1.4 Visual Inspection Prior to Use Prior to administration, vials must be visually inspect... | yes | 0.0019 |
| How should patients aged 65 or older with two conditions be handled? | - Ages 5 to 11 years: intermediate pediatric dose tier, generally half the standard adult ... | yes | 0.0019 |
| What are the absolute contraindications to the vaccine? | - Ages 5 to 11 years: intermediate pediatric dose tier, generally half the standard adult ... | yes | 0.0017 |
| When must a moderate or severe adverse event be documented? | - Prior history of a moderate reaction to an mRNA-based product, even if the reaction has ... | yes | 0.0018 |
| How long should a patient be monitored after administration? | - Prior history of a moderate reaction to an mRNA-based product, even if the reaction has ... | yes | 0.0017 |
| What should happen if the inflammatory biomarker is elevated? | - Ages 5 to 11 years: intermediate pediatric dose tier, generally half the standard adult ... | yes | 0.0018 |
