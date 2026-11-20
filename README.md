# IndicShield-RepE

**The Big Hack 2.0 · Track 03: AI Research**

**Target model:** `Qwen/Qwen2.5-3B-Instruct`

IndicShield-RepE investigates representation-level refusal steering across English, Native Kannada, Kanglish/Hinglish, and Code-Mixed + Gen-Z Slang inputs.

## Research Hypothesis

A refusal direction estimated from model activations can improve refusal consistency across Indic and code-mixed inputs when applied during prompt prefill, while threshold calibration helps preserve benign responses.

## Architecture

1. **Extract** — Estimate normalized refusal directions from refusal and non-refusal activations across model layers 12–24.
2. **Probe** — Measure prompt activations and cosine similarity to the refusal direction at the selected layer.
3. **Gate & Steer** — Compare similarity with threshold `τ`; when it exceeds the threshold, inject `+α·r` into the residual stream.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduction Commands

Run these commands from the repository root:

```powershell
python -m src.extract_vector
python -m src.run_benchmark
python results/generate_charts.py --demo
streamlit run app.py
```

`--demo` creates illustrative CSV inputs only when the corresponding CSV is missing or contains only placeholder/demo rows. Demo rows carry `status=illustrative_demo` and are not benchmark measurements. For measured charts, provide completed `results/metrics.csv`, `results/alpha_sweep.csv`, and `results/multiturn_drift.csv`, then run:

```powershell
python results/generate_charts.py
```

The chart generator exports 300-DPI `chart1_fertility.png`, `chart2_benchmark.png`, and `chart3_failure.png` into `results/`.

Measured chart inputs use these columns:

```text
metrics.csv: tier, token_fertility, mean_cosine_sim, vanilla_jsr, indicshield_jsr, over_refusal_rate
alpha_sweep.csv: alpha, jsr_pct, over_refusal_pct
multiturn_drift.csv: turn, jsr_pct, mean_cosine_sim
```

**Research scaffold status:** The benchmark CLI currently writes a `not_implemented` metrics scaffold until inference and metric collection are connected. The vector extraction entry point also requires the project’s reviewed refusal/non-refusal prompt groups and extraction workflow to be integrated before it can produce a refusal-vector artifact. The live dashboard expects `results/refusal_vec.pt` containing per-layer vectors for layers 12–24 and a selected `best_layer`; use `compute_refusal_vectors()` and `save_refusal_vectors(..., best_layer=...)` in `src/extract_vector.py` once those activations are available. Legacy single-vector helpers remain available.

## Four-Way Benchmark Table

Populate this table with measured results from the reviewed benchmark; do not report illustrative demo values as measurements.

| Linguistic tier | Vanilla JSR (%) | IndicShield-RepE JSR (%) | Over-refusal (%) | Token fertility | Mean cosine similarity |
|---|---:|---:|---:|---:|---:|
| English | — | — | — | — | — |
| Native Kannada | — | — | — | — | — |
| Kanglish / Hinglish | — | — | — | — | — |
| Code-Mixed + Gen-Z Slang | — | — | — | — | — |

Expected tier identifiers in CSV data, in chart order: `english`, `native_kannada`, `kanglish_hinglish`, `codemixed_genz_slang`.

## Team Branch Ownership

- **Researcher 1 — `feat/repe-hook`:** Representation extraction, refusal-direction methodology, and the IndicShield residual hook.
- **Researcher 2 — `feat/eval-suite`:** Safety evaluation, datasets, benchmark schemas, and compliance checks.
- **Researcher 3 — `feat/dashboard`:** Streamlit inspector, benchmark visualizations, and integration.

Shared dataset files are in `data/`; generated artifacts belong in `results/`. Coordinate before changing shared dataset schemas and use reviewed, non-operational research examples.

## Track 03 Safety Compliance

The project is scoped to the team-controlled, open-weight `Qwen/Qwen2.5-3B-Instruct` model. Harmful-response handling must preserve the evaluator’s refusal and redaction safeguards: **zero unredacted harmful outputs are published** in the dashboard, benchmark artifacts, or presentation charts.
