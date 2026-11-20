# IndicShield-RepE

**The Big Hack 2.0 · Track 03: AI Research**

IndicShield-RepE is a research scaffold for studying representation-level refusal interventions across English, Kannada, Kanglish/Hinglish, and code-mixed prompts.

**Hypothesis:** A refusal direction estimated from model activations can improve refusal consistency across Indic and code-mixed inputs when applied during prompt prefill.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The starter benchmark records are safe placeholders. Replace them only with reviewed research examples. The extraction and inference pipelines still need dataset-specific prompts and model-generation wiring.

## Run

```powershell
python -m src.run_benchmark
streamlit run app.py
```

The benchmark CLI writes a `results/metrics.csv` scaffold with `not_implemented` status until inference and metric collection are connected.

## Team ownership

- **Researcher 1 — Representation extraction:** `src/extract_vector.py`, refusal-direction methodology, and extraction validation.
- **Researcher 2 — Safety and evaluation:** `src/indicshield_hook.py`, `src/evaluator.py`, dataset schema, and safety review.
- **Researcher 3 — Benchmark and interface:** `src/run_benchmark.py`, `app.py`, metrics visualization, and integration.

Shared dataset files are in `data/`; generated artifacts belong in `results/`. Keep module interfaces stable while working in parallel, and coordinate before changing the JSON record fields.
