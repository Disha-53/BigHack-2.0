"""CLI scaffold for comparing baseline and IndicShield benchmark runs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ALPHA_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
TIERS = {"english", "native_kannada", "kanglish_hinglish", "codemixed_genz_slang"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_FIELDS = (
    "system",
    "alpha",
    "dataset",
    "example_count",
    "refusal_rate",
    "mean_token_fertility",
    "mean_latency_ms",
    "status",
)


def load_json_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSON dataset against its fixed record schema."""
    with path.open(encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a JSON array of objects in {path}.")
    schemas = {
        "indic_benchmark": {"id", "tier", "category", "is_harmful", "prompt"},
        "multiturn_drift": {"id", "tier", "turns"},
    }
    expected_keys = schemas.get(path.stem)
    if expected_keys is None:
        raise ValueError(f"No locked benchmark schema is defined for {path.stem}.")
    for index, record in enumerate(records):
        if set(record) != expected_keys:
            raise ValueError(
                f"Record {index} in {path} must have exactly these fields: "
                f"{', '.join(sorted(expected_keys))}."
            )
        if "tier" in record and (
            not isinstance(record["tier"], str) or record["tier"] not in TIERS
        ):
            raise ValueError(f"Record {index} in {path} has an unsupported tier.")
        if "turns" in record and (
            not isinstance(record["turns"], list)
            or len(record["turns"]) != 3
            or not all(isinstance(turn, str) for turn in record["turns"])
        ):
            raise ValueError(f"Record {index} in {path} must contain exactly three string turns.")
        if "is_harmful" in record and not isinstance(record["is_harmful"], bool):
            raise ValueError(f"Record {index} in {path} must use a boolean is_harmful value.")
        for field in ("id", "category", "prompt"):
            if field in record and not isinstance(record[field], str):
                raise ValueError(f"Record {index} in {path} must use a string for {field}.")
    return records


def build_placeholder_metrics(
    datasets: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Create explicitly unevaluated rows until model inference is implemented."""
    rows: list[dict[str, Any]] = []
    for dataset_name, examples in datasets.items():
        rows.append(
            {
                "system": "Vanilla",
                "alpha": "",
                "dataset": dataset_name,
                "example_count": len(examples),
                "refusal_rate": "",
                "mean_token_fertility": "",
                "mean_latency_ms": "",
                "status": "not_implemented",
            }
        )
        for alpha in ALPHA_SWEEP:
            rows.append(
                {
                    "system": "IndicShield-RepE",
                    "alpha": alpha,
                    "dataset": dataset_name,
                    "example_count": len(examples),
                    "refusal_rate": "",
                    "mean_token_fertility": "",
                    "mean_latency_ms": "",
                    "status": "not_implemented",
                }
            )
    return rows


def main() -> None:
    """Load benchmark datasets and write the baseline/alpha-sweep CSV scaffold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "metrics.csv",
        help="Path for the generated metrics CSV.",
    )
    args = parser.parse_args()

    datasets = {
        "indic_benchmark": load_json_dataset(PROJECT_ROOT / "data" / "indic_benchmark.json"),
        "multiturn_drift": load_json_dataset(PROJECT_ROOT / "data" / "multiturn_drift.json"),
    }
    rows = build_placeholder_metrics(datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} unevaluated metric rows to {args.output}")


if __name__ == "__main__":
    main()
