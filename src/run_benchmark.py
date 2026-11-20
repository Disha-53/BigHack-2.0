"""Run deterministic Vanilla and IndicShield-RepE refusal benchmarks."""

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluator import compute_token_fertility, is_refusal, redact_if_unsafe
from src.indicshield_hook import IndicShieldHook

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ALPHA_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
TIERS = ("english", "native_kannada", "kanglish_hinglish", "codemixed_genz_slang")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = (
    "tier",
    "token_fertility",
    "mean_cosine_sim",
    "vanilla_jsr",
    "indicshield_jsr",
    "over_refusal_rate",
    "avg_latency_ms",
)
LOG_COLUMNS = (
    "id",
    "tier",
    "category",
    "is_harmful",
    "prompt",
    "vanilla_response",
    "vanilla_refused",
    "indicshield_response",
    "indicshield_refused",
    "cosine_sim",
    "latency_ms",
    "token_fertility",
)
ALPHA_COLUMNS = ("alpha", "jsr_pct", "over_refusal_pct")
MULTITURN_COLUMNS = ("turn", "jsr_pct", "mean_cosine_sim")
BENCHMARK_FIELDS = {"id", "tier", "category", "is_harmful", "prompt"}
MULTITURN_FIELDS = {"id", "tier", "turns"}


class CosineMonitor:
    """Collect pre-intervention last-token similarity values at one model layer."""

    def __init__(self, refusal_vec: torch.Tensor) -> None:
        self.refusal_vec = refusal_vec.detach()
        self.values: list[float] = []

    def __call__(self, module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        """Record cosine similarity for multi-token prefill activations."""
        del module, inputs
        if isinstance(output, torch.Tensor):
            activations = output
        elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            activations = output[0]
        else:
            raise TypeError("Monitored layer output must start with a tensor.")

        if activations.ndim != 3 or activations.shape[1] <= 1:
            return

        vector = self.refusal_vec.to(device=activations.device, dtype=torch.float32)
        if vector.ndim != 1 or vector.shape[0] != activations.shape[-1]:
            raise ValueError("Refusal vector must match the model residual hidden size.")
        similarities = torch.nn.functional.cosine_similarity(
            activations[:, -1, :].float(),
            vector.expand(activations.shape[0], -1),
            dim=-1,
        )
        self.values.append(float(similarities.mean().item()))


def _resolve_path(path: Path) -> Path:
    """Resolve relative CLI paths from the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json_records(path: Path, fields: set[str]) -> list[dict[str, Any]]:
    """Load a list of records and enforce its exact top-level record schema."""
    with path.open(encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a JSON array of objects in {path}.")
    for index, record in enumerate(records):
        if set(record) != fields:
            raise ValueError(
                f"Record {index} in {path} must have exactly these fields: "
                f"{', '.join(sorted(fields))}."
            )
    return records


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Load and validate the Indic benchmark records."""
    records = _load_json_records(path, BENCHMARK_FIELDS)
    for index, record in enumerate(records):
        if not isinstance(record["tier"], str) or record["tier"] not in TIERS:
            raise ValueError(f"Record {index} in {path} has an unsupported tier.")
        if not isinstance(record["id"], str) or not isinstance(record["category"], str):
            raise ValueError(f"Record {index} in {path} must have string IDs and categories.")
        if not isinstance(record["is_harmful"], bool) or not isinstance(record["prompt"], str):
            raise ValueError(f"Record {index} in {path} has invalid harmful flag or prompt.")
    if not records:
        raise ValueError(f"Benchmark dataset {path} must not be empty.")
    return records


def load_multiturn(path: Path) -> list[dict[str, Any]]:
    """Load and validate three-turn drift conversations."""
    records = _load_json_records(path, MULTITURN_FIELDS)
    if not records:
        raise ValueError(f"Multi-turn dataset {path} must not be empty.")
    for index, record in enumerate(records):
        turns = record["turns"]
        if not isinstance(record["tier"], str) or record["tier"] not in TIERS:
            raise ValueError(f"Record {index} in {path} has an unsupported tier.")
        if (
            not isinstance(record["id"], str)
            or not isinstance(turns, list)
            or len(turns) != 3
            or not all(isinstance(turn, str) for turn in turns)
        ):
            raise ValueError(f"Record {index} in {path} must contain an ID and exactly 3 strings.")
    return records


def load_model(model_name: str) -> tuple[Any, Any]:
    """Load an evaluation model in float16 and put it in inference mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_refusal_vector(path: Path) -> tuple[int, torch.Tensor]:
    """Load the selected layer and its refusal direction from a saved checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "best_layer" not in checkpoint or "vectors" not in checkpoint:
        raise ValueError("Refusal-vector checkpoint must contain 'best_layer' and 'vectors'.")
    best_layer = checkpoint["best_layer"]
    vectors = checkpoint["vectors"]
    if not isinstance(best_layer, int) or isinstance(best_layer, bool):
        raise ValueError("Checkpoint 'best_layer' must be an integer.")
    if not isinstance(vectors, Mapping):
        raise ValueError("Checkpoint 'vectors' must map layer indices to tensors.")
    vector = vectors.get(best_layer)
    if vector is None:
        vector = vectors.get(str(best_layer))
    if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
        raise ValueError(f"Checkpoint has no 1-D refusal vector for layer {best_layer}.")
    return best_layer, vector


def get_decoder_layer(model: Any, layer_idx: int) -> torch.nn.Module:
    """Return a decoder block by index, reporting unsupported model layouts."""
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("The loaded model does not expose decoder layers at model.layers.")
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Refusal-vector layer {layer_idx} is outside the model's decoder layers.")
    return layers[layer_idx]


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Generate a greedy chat response using the tokenizer's chat template."""
    chat_prompt = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(chat_prompt, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    prompt_length = encoded["input_ids"].shape[1]
    return tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)


def _rate_summary(
    records: Sequence[dict[str, Any]],
    refusals: Sequence[bool],
) -> tuple[float, float]:
    """Return harmful jailbreak-success and benign over-refusal rates."""
    harmful = [not refused for row, refused in zip(records, refusals) if row["is_harmful"]]
    benign = [refused for row, refused in zip(records, refusals) if not row["is_harmful"]]
    jsr = sum(harmful) / len(harmful) if harmful else math.nan
    over_refusal = sum(benign) / len(benign) if benign else math.nan
    return jsr, over_refusal


def _mean_or_nan(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or NaN when no observations exist."""
    return mean(values) if values else math.nan


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write rows using a stable, explicitly specified CSV column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _run_benchmark(
    model: Any,
    tokenizer: Any,
    hook: IndicShieldHook,
    monitor: CosineMonitor,
    records: list[dict[str, Any]],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[bool], list[bool]]:
    """Generate Vanilla and shielded responses and collect per-prompt diagnostics."""
    logs: list[dict[str, Any]] = []
    vanilla_refusals: list[bool] = []
    shield_refusals: list[bool] = []
    for record in records:
        messages = [{"role": "user", "content": record["prompt"]}]
        vanilla_response = generate_response(model, tokenizer, messages, max_new_tokens)
        vanilla_refused = is_refusal(vanilla_response)

        similarity_start = len(monitor.values)
        latency_start = len(hook.latencies_ms)
        shield_response = generate_response(model, tokenizer, messages, max_new_tokens)
        shield_refused = is_refusal(shield_response)
        similarities = monitor.values[similarity_start:]
        latencies = hook.latencies_ms[latency_start:]
        similarity = _mean_or_nan(similarities)
        latency = _mean_or_nan(latencies)

        vanilla_refusals.append(vanilla_refused)
        shield_refusals.append(shield_refused)
        logs.append(
            {
                "id": record["id"],
                "tier": record["tier"],
                "category": record["category"],
                "is_harmful": record["is_harmful"],
                "prompt": record["prompt"],
                "vanilla_response": redact_if_unsafe(
                    vanilla_response, record["is_harmful"], vanilla_refused
                ),
                "vanilla_refused": vanilla_refused,
                "indicshield_response": redact_if_unsafe(
                    shield_response, record["is_harmful"], shield_refused
                ),
                "indicshield_refused": shield_refused,
                "cosine_sim": similarity,
                "latency_ms": latency,
                "token_fertility": compute_token_fertility(tokenizer, record["prompt"]),
            }
        )
    return logs, vanilla_refusals, shield_refusals


def _run_alpha_sweep(
    model: Any,
    tokenizer: Any,
    hook: IndicShieldHook,
    records: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Evaluate refusal and benign over-refusal rates at each configured alpha."""
    rows: list[dict[str, Any]] = []
    for alpha in ALPHA_SWEEP:
        hook.alpha = alpha
        refusals = [
            is_refusal(
                generate_response(
                    model,
                    tokenizer,
                    [{"role": "user", "content": record["prompt"]}],
                    max_new_tokens,
                )
            )
            for record in records
        ]
        jsr, over_refusal = _rate_summary(records, refusals)
        rows.append(
            {
                "alpha": alpha,
                "jsr_pct": jsr * 100,
                "over_refusal_pct": over_refusal * 100,
            }
        )
    return rows


def _run_multiturn(
    model: Any,
    tokenizer: Any,
    monitor: CosineMonitor,
    conversations: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Measure refusal after zero, one, or two benign conversational context turns."""
    rows: list[dict[str, Any]] = []
    for turn_count in (1, 2, 3):
        refusals: list[bool] = []
        similarities: list[float] = []
        for conversation in conversations:
            turns = conversation["turns"]
            messages: list[dict[str, str]] = []
            if turn_count >= 2:
                messages.append({"role": "user", "content": turns[0]})
            if turn_count >= 3:
                messages.append({"role": "assistant", "content": turns[1]})
            messages.append({"role": "user", "content": turns[2]})

            similarity_start = len(monitor.values)
            response = generate_response(model, tokenizer, messages, max_new_tokens)
            refusals.append(is_refusal(response))
            similarities.extend(monitor.values[similarity_start:])

        rows.append(
            {
                "turn": turn_count,
                "jsr_pct": 100 * sum(not refused for refused in refusals) / len(refusals),
                "mean_cosine_sim": _mean_or_nan(similarities),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for model, checkpoint, and benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--vec_path", type=Path, default=Path("results/refusal_vec.pt"))
    parser.add_argument("--benchmark_path", type=Path, default=Path("data/indic_benchmark.json"))
    parser.add_argument("--multiturn_path", type=Path, default=Path("data/multiturn_drift.json"))
    parser.add_argument("--tau", type=float, default=0.65)
    parser.add_argument("--alpha", type=float, default=1.6)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser


def main() -> None:
    """Run the full benchmark and export the four requested result CSVs."""
    args = build_parser().parse_args()
    benchmark_path = _resolve_path(args.benchmark_path)
    multiturn_path = _resolve_path(args.multiturn_path)
    vec_path = _resolve_path(args.vec_path)
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be a positive integer.")

    records = load_benchmark(benchmark_path)
    conversations = load_multiturn(multiturn_path)
    best_layer, refusal_vec = load_refusal_vector(vec_path)
    model, tokenizer = load_model(args.model_name)
    layer = get_decoder_layer(model, best_layer)

    monitor = CosineMonitor(refusal_vec)
    monitor_handle = layer.register_forward_hook(monitor)
    hook = IndicShieldHook(refusal_vec, best_layer, args.tau, args.alpha)
    hook.attach(layer)
    try:
        logs, vanilla_refusals, shield_refusals = _run_benchmark(
            model, tokenizer, hook, monitor, records, args.max_new_tokens
        )
        alpha_rows = _run_alpha_sweep(
            model, tokenizer, hook, records, args.max_new_tokens
        )
        hook.alpha = args.alpha
        multiturn_rows = _run_multiturn(
            model, tokenizer, monitor, conversations, args.max_new_tokens
        )
    finally:
        hook.remove()
        monitor_handle.remove()

    metric_rows: list[dict[str, Any]] = []
    for tier in TIERS:
        indices = [index for index, record in enumerate(records) if record["tier"] == tier]
        tier_records = [records[index] for index in indices]
        vanilla_jsr, _ = _rate_summary(
            tier_records, [vanilla_refusals[index] for index in indices]
        )
        shield_jsr, over_refusal = _rate_summary(
            tier_records, [shield_refusals[index] for index in indices]
        )
        tier_logs = [logs[index] for index in indices]
        metric_rows.append(
            {
                "tier": tier,
                "token_fertility": _mean_or_nan(
                    [row["token_fertility"] for row in tier_logs]
                ),
                "mean_cosine_sim": _mean_or_nan(
                    [row["cosine_sim"] for row in tier_logs if not math.isnan(row["cosine_sim"])]
                ),
                "vanilla_jsr": vanilla_jsr,
                "indicshield_jsr": shield_jsr,
                "over_refusal_rate": over_refusal,
                "avg_latency_ms": _mean_or_nan(
                    [row["latency_ms"] for row in tier_logs if not math.isnan(row["latency_ms"])]
                ),
            }
        )

    _write_csv(PROJECT_ROOT / "results" / "metrics.csv", METRIC_COLUMNS, metric_rows)
    _write_csv(PROJECT_ROOT / "results" / "eval_logs.csv", LOG_COLUMNS, logs)
    _write_csv(PROJECT_ROOT / "results" / "alpha_sweep.csv", ALPHA_COLUMNS, alpha_rows)
    _write_csv(PROJECT_ROOT / "results" / "multiturn_drift.csv", MULTITURN_COLUMNS, multiturn_rows)
    print("Wrote metrics.csv, eval_logs.csv, alpha_sweep.csv, and multiturn_drift.csv to results/.")


if __name__ == "__main__":
    main()
