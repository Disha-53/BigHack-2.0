# IndicShield-RepE

**The Big Hack 2.0 · Track 03: AI Research**

**Target model:** `Qwen/Qwen2.5-3B-Instruct`

## Research hypothesis

**A refusal direction estimated from model activations can improve refusal consistency across Indic and code-mixed inputs when applied during prompt prefill, while threshold calibration helps preserve benign responses.**

## Three-step research pipeline

1. **Extract** — Estimate normalized refusal directions from harmful/harmless prompt-pair activations across layers 12–24.
2. **Probe** — Measure last-token activation similarity to the refusal direction at the selected layer.
3. **Gate & Steer** — Compare cosine similarity with `tau`; when similarity meets or exceeds the threshold, add `alpha * refusal_vector` to the residual stream.

The local, git-ignored `results/refusal_vec.pt` checkpoint identifies `Qwen/Qwen2.5-3B-Instruct`, selects layer 19, and records AUROC `1.0000` for that layer. The checkpoint also reports `1.0000` for layers 20–24. This is the checkpoint's stored extraction result, not an independently reproduced result or evidence of cross-dataset generalization; the raw activations and split manifest are not present in the repository.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## CLI commands

Run from the repository root:

```powershell
python src/extract_vector.py
python src/run_benchmark.py
python results/generate_charts.py
streamlit run app.py
```

`src/extract_vector.py` reads paired harmful/harmless prompts from `data/english_pairs.json`, probes layers 12–24, computes per-layer refusal directions and AUROC, and saves the vectors and selected layer to `results/refusal_vec.pt`. Optional `--model_name`, `--data_path`, `--output_path`, and `--layers` arguments override its defaults. The benchmark command requires a compatible checkpoint and locally available model weights (or access to download them).

The Streamlit threshold slider defaults to `0.18` and recalculates its live per-prompt gate status and calibration curve from `results/eval_logs.csv`; the benchmark CLI's independent `--tau` default remains `0.65`.

`results/generate_charts.py` reads `metrics.csv`, `alpha_sweep.csv`, and `multiturn_drift.csv`; it fails with a diagnostic if required inputs are absent or empty. To inspect layout without benchmark results, run:

```powershell
python results/generate_charts.py --demo
```

Demo data are watermarked **ILLUSTRATIVE PREVIEW — NOT FINAL OR MEASURED** and are never substituted silently. Charts are exported at 300 DPI to `results/chart1_fertility.png`, `results/chart2_benchmark.png`, and `results/chart3_failure.png`. CSV charts are labeled as local inputs whose provenance has not been independently verified.

The Streamlit inspector also renders interactive Altair calibration, tier, fertility, multi-turn, and alpha charts from the local result CSVs; changing its `tau` slider updates the gate badge and threshold metrics without running inference. It starts in safe preview mode if the local refusal vector, hook, cached model weights, or CUDA GPU is unavailable. Preview mode does not produce model responses or invented metric values. In live mode, unreviewed prompts are treated as harmful and every response not detected as a refusal is displayed only as `[REDACTED - UNSAFE RESPONSE]`.

## Current result snapshot and limitations

The local result CSVs are ignored/generated artifacts, not tracked benchmark deliverables. They contain 4 tier rows in `metrics.csv`, 6 alpha rows, 3 multi-turn rows, and 80 per-prompt log rows. The CSVs do not contain a run manifest, seed, hardware, or provenance metadata, so the numbers below describe the current local snapshot only.

- **Threshold (`tau`):** The benchmark default is `0.65`; the current CSVs do not include a threshold sweep, so they do not establish that this value is optimal.
- **Steering strength (`alpha`):** The benchmark default is `1.6`. The local alpha sweep reports 65% JSR and 2.5% over-refusal at each tested value from 0.5 to 3.0. The flat sweep does not demonstrate a measured benefit from changing alpha.
- **Per-tier JSR:** The current CSV reports identical Vanilla and IndicShield-RepE rates: English 50%, Native Kannada 100%, Kanglish/Hinglish 60%, and Code-Mixed + Gen-Z Slang 50%. Do not describe this snapshot as an observed JSR improvement.
- **Multi-turn drift:** The current CSV reports JSR of approximately 58.3%, 66.7%, and 66.7% at Turns 1–3, while mean cosine similarity decreases from about 0.212 to 0.189. This is a descriptive local run, not causal evidence.
- **Latency:** The current metrics CSV reports 0.0 ms average hook latency for every tier; treat that field as unavailable/unreliable, not as measured zero-cost inference.
- **30-sample audit:** A 30-sample manual audit is not accompanied by a sample manifest, rubric, or audit results in `data/` or `results/`. It cannot be independently verified and must not be presented as a held-out evaluation or as a substitute for the 80-record benchmark.

### Track 03 safety compliance

Harmful research prompts are abstract and contain no operational instructions or personal data. Before publication, unsafe outputs must remain masked as `[REDACTED - UNSAFE RESPONSE]`. A check of the current local `results/eval_logs.csv` found 80 rows, including 40 harmful prompts; both Vanilla and IndicShield columns had **zero unrefused harmful responses left unredacted**. This verifies the current local log only; logs and the refusal-vector checkpoint are git-ignored and are not included in a clone by default.

## Team branch ownership

- **Member 1 — `feat/repe-hook`:** Representation extraction, refusal-direction methodology, and the IndicShield residual hook.
- **Member 2 — `feat/eval-suite`:** Safety evaluator, benchmark datasets/schemas, and benchmark runner.
- **Member 3 — `feat/dashboard`:** Streamlit research inspector, result visualizations, and integration.

Shared inputs are in `data/`; generated artifacts belong in `results/`. Coordinate before changing dataset schemas, and do not publish illustrative preview data as measurements.
