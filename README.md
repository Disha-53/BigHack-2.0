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
python -m src.extract_vector
streamlit run app.py
```

The benchmark CLI writes a `results/metrics.csv` scaffold with `not_implemented` status until inference and metric collection are connected.
The extraction CLI reads harmful/harmless prompt pairs from `data/english_pairs.json` and saves normalized layer-wise vectors, AUROC scores, and the best layer to `results/refusal_vec.pt`. Override the model, data, output, or probed layers with `--model_name`, `--data_path`, `--output_path`, and `--layers`. The starter prompt pairs are placeholders, so replace them with reviewed research examples before interpreting extraction results.
`src.indicshield_hook.IndicShieldHook` attaches to the selected decoder layer with `with hook.attach(model):` and conditionally steers prefill activations when their last-token cosine similarity exceeds `tau`. Its `get_stats()` method reports the latest gate decision and measured hook-call latency; latency depends on the model, device, and prompt.

## Team ownership

- **Researcher 1 — Representation extraction:** `src/extract_vector.py`, refusal-direction methodology, and extraction validation.
- **Researcher 2 — Safety and evaluation:** `src/indicshield_hook.py`, `src/evaluator.py`, dataset schema, and safety review.
- **Researcher 3 — Benchmark and interface:** `src/run_benchmark.py`, `app.py`, metrics visualization, and integration.

Shared dataset files are in `data/`; generated artifacts belong in `results/`. Keep module interfaces stable while working in parallel, and coordinate before changing the JSON record fields.
