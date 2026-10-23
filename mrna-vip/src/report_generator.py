"""Module 4 - LLM-powered, schema-validated clinical safety report.

Flow for one patient
    1. Module 1 scores the patient (risk label + probability).
    2. Module 3 retrieves the guideline chunks relevant to the patient's profile.
    3. A prompt is built from ONLY those two things (+ the patient's own numbers).
    4. The LLM answers; the answer is parsed + validated against ClinicalSafetyReport
       (one retry on failure - FR-4.3).
    5. Guardrails re-check the answer against the retrieved text (FR-4.4, safety constraints).
    6. The report is saved as .json and .md (FR-4.6).

LLM choice (env LLM_PROVIDER = openai | anthropic | offline | auto):
    auto -> OpenAI if OPENAI_API_KEY is set, else Anthropic if ANTHROPIC_API_KEY is set,
            else the OFFLINE rule-based generator (no API key needed, deterministic).
The provider that really produced each report is written inside the .md file.
"""
import argparse
import json
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from src import guideline_rules as gr
from src.config import (
    CLEAN_CSV, ELEVATED_BIOMARKER_MGL, OUTPUT_DIR, REPORT_DIR, REQUIRED_DISCLAIMER,
)
from src.data_pipeline import load_clean
from src.rag_index import GuidelineIndex, build_index, load_index
from src.risk_model import load_model, score_patient
from src.schemas import ClinicalSafetyReport

load_dotenv()

SYSTEM_PROMPT = """You are a clinical safety reporting assistant for a STUDENT ENGINEERING PROJECT.
You may ONLY use (a) the predicted risk label/probability and (b) the guideline excerpts supplied below.

STRICT RULES
1. Never recommend a dose that contradicts the guideline excerpts.
2. Never invent a clinical metric, statistic or guideline that is not present in the excerpts.
3. If the excerpts do not contain enough information to decide confidently, set Eligible_For_Vaccine
   to false, set Recommended_Dose_mg to 0, and explain why in Reasoning_Steps instead of guessing.
4. Every report must keep the Disclaimer field: it states that this is a student engineering
   project and not real clinical guidance.

OUTPUT NOTES
- Doses in the patient summary are given in micrograms (mcg) AND milligrams (mg). Recommended_Dose_mg is in MILLIGRAMS (1 mg = 1000 mcg).
- If the patient needs secondary clinician review, Eligible_For_Vaccine is false and Recommended_Dose_mg is 0.
- FDA_Safety_Warnings_Cited must contain sentences copied WORD FOR WORD from the excerpts.
- Return ONLY the JSON object described below.
{format_instructions}"""

USER_PROMPT = """Patient summary: {patient_summary}
Predicted adverse-reaction risk: {risk_label} (probability {risk_probability:.2f})

Relevant guideline excerpts:
{guidelines}

Generate the structured clinical safety report now."""


class ReportGenerationError(RuntimeError):
    """The LLM answer failed schema validation twice (initial attempt + the single retry)."""


# ----------------------------------------------------------------------------- retrieval
def build_queries(p: gr.Profile) -> list[str]:
    return [
        f"Dosing guidance for a patient aged {p.age}",
        f"Candidate dose of {p.dose_mcg:g} micrograms and secondary clinician review",
        "Dose adjustment for an elevated inflammatory biomarker level",
        "Precautions requiring secondary review: pre-existing conditions and prior reactions",
        "Monitoring after administration and adverse event reporting",
    ]


def retrieve_for_patient(index: GuidelineIndex, p: gr.Profile, k: int = 3):
    """Run several profile-specific queries; return unique chunks (first-seen order) + total latency."""
    seen, chunks, total = set(), [], 0.0
    for q in build_queries(p):
        got, latency = index.query(q, k)
        total += latency
        for c in got:
            if c not in seen:
                seen.add(c)
                chunks.append(c)
    return chunks, total


# ----------------------------------------------------------------------------- LLM back-ends
def offline_draft(p: gr.Profile, chunks: list[str]) -> dict:
    """Deterministic stand-in for an LLM: builds the draft from the guideline rules."""
    d = gr.decide(p, chunks)
    steps = [f"Risk model output: {p.risk_label} risk of an adverse reaction (probability {p.risk_probability:.2f})."]
    steps.append(f"Candidate dose {p.dose_mcg:g} mcg = {p.dose_mg:.4f} mg; patient age {p.age}; "
                 f"{p.conditions} pre-existing condition(s); prior reaction history: {p.prior_reaction}.")
    cited = []
    for f in d["triggered"]:
        if f.snippet_found:
            steps.append(f"Guideline rule triggered - {f.rule.description}: secondary clinician review is required.")
            cited.append(f.rule.snippet)
        else:
            steps.append(f"Rule '{f.rule.description}' seems to apply, but the retrieved excerpts do not contain the "
                         "supporting guideline text, so a conservative default is used.")
    if d["high_ml_risk"] and not d["triggered"]:
        steps.append("No guideline review trigger found, but the risk model predicts HIGH risk, so the conservative "
                     "choice is to defer to a clinician.")
    if gr.snippet_in_chunks(gr.MONITORING_SNIPPET, chunks):
        cited.append(gr.MONITORING_SNIPPET)
    eligible = d["eligible"] and not d["missing_support"]
    if eligible:
        steps.append("No guideline review trigger and a LOW predicted risk: eligible under standard administration, "
                     "with the standard post-administration monitoring.")
        dose = p.dose_mg
    else:
        steps.append("Decision: NOT eligible for standard administration pending secondary clinician review; "
                     "no dose is recommended until then.")
        dose = 0.0
    return {"Eligible_For_Vaccine": eligible, "Recommended_Dose_mg": dose, "Reasoning_Steps": steps,
            "FDA_Safety_Warnings_Cited": cited, "Disclaimer": REQUIRED_DISCLAIMER}


def get_llm(provider: str | None = None):
    """Return (runnable_or_None, label). None means: use the offline rule-based generator."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()
    if provider in ("openai", "auto") and os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0), f"openai:{model}"
    if provider in ("anthropic", "auto") and os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
        return ChatAnthropic(model=model, temperature=0), f"anthropic:{model}"
    if provider in ("openai", "anthropic"):
        raise RuntimeError(f"LLM_PROVIDER={provider} but its API key is not set in .env")
    return None, "offline-rule-based (no API key)"


# ----------------------------------------------------------------------------- guardrails
def apply_guardrails(report: ClinicalSafetyReport, p: gr.Profile, chunks: list[str]):
    """Independent checks on ANY draft (real LLM or offline). Returns (report, notes)."""
    notes, eligible, dose = [], report.Eligible_For_Vaccine, report.Recommended_Dose_mg
    steps = list(report.Reasoning_Steps)

    # (a) citations must be real sentences from the retrieved excerpts (no invented guidelines)
    kept = [c for c in report.FDA_Safety_Warnings_Cited if gr.snippet_in_chunks(c, chunks)]
    dropped = len(report.FDA_Safety_Warnings_Cited) - len(kept)
    if dropped:
        notes.append(f"Removed {dropped} citation(s) that do not appear verbatim in the retrieved guideline text.")
    if not kept:
        notes.append("No verifiable guideline citation remains; conservative default applied.")
        eligible, dose = False, 0.0

    # (b) FR-4.4: never recommend a dose above the bound stated in the retrieved guideline
    bound = gr.parse_dose_bound_mcg(chunks)
    if bound is not None and dose * 1000.0 > bound + 1e-9:
        notes.append(f"Recommended dose {dose:.4f} mg exceeds the guideline bound of {bound:g} mcg "
                     f"({bound / 1000:.3f} mg) - rejected, patient marked not eligible.")
        eligible, dose = False, 0.0

    # (c) a draft may not approve a patient the guideline rules say needs review
    d = gr.decide(p, chunks)
    if eligible and (d["triggered"] or d["high_ml_risk"]):
        why = ", ".join(f.rule.description for f in d["triggered"]) or "predicted HIGH risk"
        notes.append(f"Draft marked the patient eligible, but review is required ({why}); overridden.")
        eligible, dose = False, 0.0

    if not eligible:
        dose = 0.0
    if notes:
        steps += [f"GUARDRAIL: {n}" for n in notes]
    fixed = ClinicalSafetyReport(Eligible_For_Vaccine=eligible, Recommended_Dose_mg=dose, Reasoning_Steps=steps,
                                 FDA_Safety_Warnings_Cited=kept, Disclaimer=report.Disclaimer)
    return fixed, notes


# ----------------------------------------------------------------------------- generation
def _invoke_with_one_retry(chain, inputs):
    """Validate the LLM answer; on failure retry EXACTLY once (FR-4.3), then raise."""
    last = None
    for attempt in (1, 2):
        try:
            return chain.invoke(inputs), attempt
        except (OutputParserException, ValidationError) as exc:
            last = exc
    raise ReportGenerationError(
        "LLM output failed schema validation on both attempts (1 initial + 1 retry): " + str(last)[:300]) from last


@dataclass
class Generated:
    report: ClinicalSafetyReport
    attempts: int
    llm_latency_s: float
    notes: list
    provider: str


def generate_report(p: gr.Profile, chunks: list[str], llm=None, provider_label: str | None = None) -> Generated:
    parser = PydanticOutputParser(pydantic_object=ClinicalSafetyReport)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", USER_PROMPT)])
    if llm is None:
        llm = RunnableLambda(lambda _pv: AIMessage(content=json.dumps(offline_draft(p, chunks))))
        provider_label = provider_label or "offline-rule-based (no API key)"
    chain = prompt | llm | parser
    inputs = {
        "format_instructions": parser.get_format_instructions(),
        "patient_summary": p.summary(),
        "risk_label": p.risk_label,
        "risk_probability": p.risk_probability,
        "guidelines": "\n---\n".join(chunks),
    }
    start = time.perf_counter()
    draft, attempts = _invoke_with_one_retry(chain, inputs)
    latency = time.perf_counter() - start
    final, notes = apply_guardrails(draft, p, chunks)
    return Generated(final, attempts, latency, notes, provider_label or "custom")


# ----------------------------------------------------------------------------- persistence
def report_markdown(p: gr.Profile, g: Generated, chunks_backend: str) -> str:
    r = g.report
    lines = [
        f"> **{r.Disclaimer}**", "",
        f"# Clinical Safety Report - patient {p.patient_id}", "",
        "## Patient and risk score", f"- {p.summary()}",
        f"- Predicted adverse-reaction risk: **{p.risk_label}** (probability {p.risk_probability:.2f})", "",
        "## Decision",
        f"- Eligible for vaccine (standard administration): **{'YES' if r.Eligible_For_Vaccine else 'NO - clinician review required'}**",
        f"- Recommended dose: **{r.Recommended_Dose_mg:.4f} mg** ({r.Recommended_Dose_mg * 1000:g} mcg)", "",
        "## Reasoning steps",
    ]
    lines += [f"{i}. {s}" for i, s in enumerate(r.Reasoning_Steps, 1)]
    lines += ["", "## Guideline excerpts cited (verbatim)"]
    lines += [f"- \"{c}\"" for c in r.FDA_Safety_Warnings_Cited] or ["- (none)"]
    lines += ["", "## Provenance",
              f"- Report drafted by: `{g.provider}` (attempts: {g.attempts}, latency {g.llm_latency_s:.3f}s)",
              f"- Retrieval embedding backend: `{chunks_backend}`",
              f"- Elevated-biomarker cut-off used: {ELEVATED_BIOMARKER_MGL:g} mg/L (project assumption)", "",
              f"*{r.Disclaimer}*", ""]
    return "\n".join(lines)


def save_report(p: gr.Profile, g: Generated, backend: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    jp, mp = REPORT_DIR / f"report_{p.patient_id}.json", REPORT_DIR / f"report_{p.patient_id}.md"
    jp.write_text(json.dumps(g.report.model_dump(), indent=2))
    mp.write_text(report_markdown(p, g, backend), encoding="utf-8")
    return jp, mp


# ----------------------------------------------------------------------------- orchestration
def make_profile(row: dict, model) -> gr.Profile:
    label, prob = score_patient(model, row)
    return gr.Profile(patient_id=row["Patient_ID"], age=int(row["Age"]), conditions=int(row["Pre_Existing_Conditions"]),
                      dose_mcg=float(row["Vaccine_Dose_mcg"]), biomarker=float(row["Biomarker_Level"]),
                      prior_reaction=str(row["Prior_Reaction_History"]), risk_label=label, risk_probability=prob)


def pick_sample_patients(df) -> list[str]:
    """Five patients covering different guideline situations (deterministic: first match in file order)."""
    conds = {
        "no trigger": lambda r: r.Vaccine_Dose_mcg <= 100 and r.Pre_Existing_Conditions <= 1 and r.Prior_Reaction_History == "None" and r.Biomarker_Level < ELEVATED_BIOMARKER_MGL and r.Age < 65,
        "dose over 100": lambda r: r.Vaccine_Dose_mcg > 100 and r.Biomarker_Level < ELEVATED_BIOMARKER_MGL and r.Pre_Existing_Conditions <= 1,
        "age 65+ with 2+ conditions": lambda r: r.Age >= 65 and r.Pre_Existing_Conditions >= 2,
        "prior moderate reaction": lambda r: r.Prior_Reaction_History == "Moderate",
        "dose over 100 + elevated biomarker": lambda r: r.Vaccine_Dose_mcg > 100 and r.Biomarker_Level >= ELEVATED_BIOMARKER_MGL,
    }
    picks = []
    for fn in conds.values():
        match = df[df.apply(fn, axis=1)]
        if len(match):
            picks.append(match.iloc[0]["Patient_ID"])
    return picks


def run(patient_ids: list[str] | None = None):
    df = load_clean()
    model = load_model()
    try:
        index = load_index()
    except Exception:
        index = build_index()
    llm, label = get_llm()
    ids = patient_ids or pick_sample_patients(df)
    rag_times, llm_times = [], []
    print(f"Generating {len(ids)} report(s) with: {label} | retrieval backend: {index.backend}")
    for pid in ids:
        row = df[df.Patient_ID == pid].iloc[0].to_dict()
        p = make_profile(row, model)
        chunks, rag_latency = retrieve_for_patient(index, p)
        g = generate_report(p, chunks, llm=llm, provider_label=label)
        save_report(p, g, index.backend)
        rag_times.append(rag_latency)
        llm_times.append(g.llm_latency_s)
        print(f"  {pid}: risk={p.risk_label} ({p.risk_probability:.2f}) -> eligible={g.report.Eligible_For_Vaccine}, "
              f"dose={g.report.Recommended_Dose_mg:.4f} mg, attempts={g.attempts}")
    (OUTPUT_DIR / "latencies.json").write_text(json.dumps({
        "rag_retrieval_s": sum(rag_times) / len(rag_times),
        "report_generation_s": sum(llm_times) / len(llm_times),
        "provider": label, "retrieval_backend": index.backend, "n_reports": len(ids)}, indent=2))
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate clinical safety reports")
    ap.add_argument("--patient", nargs="*", help="Patient IDs, e.g. P0003 P0042 (default: 5 diverse samples)")
    run(ap.parse_args().patient)
