"""Generate dark-theme benchmark charts from result CSVs, with explicit preview mode."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent
TIERS = ("english", "native_kannada", "kanglish_hinglish", "codemixed_genz_slang")
TIER_LABELS = ("English", "Native Kannada", "Kanglish / Hinglish", "Code-Mixed + Gen-Z")
METRICS_COLUMNS = (
    "tier",
    "token_fertility",
    "mean_cosine_sim",
    "vanilla_jsr",
    "indicshield_jsr",
    "over_refusal_rate",
    "avg_latency_ms",
)
ALPHA_COLUMNS = ("alpha", "jsr_pct", "over_refusal_pct")
MULTITURN_COLUMNS = ("turn", "jsr_pct", "mean_cosine_sim")
COLORS = {
    "background": "#0F172A",
    "panel": "#1E293B",
    "cyan": "#38BDF8",
    "emerald": "#10B981",
    "coral": "#F43F5E",
    "amber": "#F59E0B",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "grid": "#334155",
}


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]] | None:
    """Return CSV records, or None when the file is absent or has no data rows."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return None
        if tuple(reader.fieldnames) != columns:
            raise ValueError(
                f"{path.name} must have exactly these columns: {', '.join(columns)}."
            )
        rows = list(reader)
    return rows or None


def _number(row: dict[str, str], column: str, source: str) -> float:
    """Parse a finite numeric value from a CSV record."""
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source} has a missing or non-numeric {column!r} value.") from error
    if not math.isfinite(value):
        raise ValueError(f"{source} has a non-finite {column!r} value.")
    return value


def _percentage(value: float) -> float:
    """Normalize a fractional rate or a percentage to a 0–100 scale."""
    return value * 100 if 0 <= value <= 1 else value


def _validate_metrics(rows: list[dict[str, str]], source: str) -> list[dict[str, Any]]:
    """Validate benchmark metric rows and arrange them in the required tier order."""
    if any(row.get("status", "").casefold() in {"not_implemented", "illustrative_demo"} for row in rows):
        raise ValueError(
            f"{source} contains placeholder/demo rows; do not chart them as measured results."
        )
    by_tier: dict[str, dict[str, Any]] = {}
    for row in rows:
        tier = row.get("tier", "")
        if tier not in TIERS:
            raise ValueError(f"{source} contains unsupported tier {tier!r}.")
        if tier in by_tier:
            raise ValueError(f"{source} contains duplicate rows for tier {tier!r}.")
        parsed: dict[str, Any] = {"tier": tier}
        parsed["token_fertility"] = _number(row, "token_fertility", source)
        parsed["mean_cosine_sim"] = _number(row, "mean_cosine_sim", source)
        for column in ("vanilla_jsr", "indicshield_jsr", "over_refusal_rate"):
            parsed[column] = _percentage(_number(row, column, source))
            if not 0 <= parsed[column] <= 100:
                raise ValueError(f"{source} has {column} outside the 0–100 range.")
        if _number(row, "avg_latency_ms", source) < 0:
            raise ValueError(f"{source} has negative average hook latency.")
        if parsed["token_fertility"] < 0:
            raise ValueError(f"{source} has negative token fertility.")
        if not -1 <= parsed["mean_cosine_sim"] <= 1:
            raise ValueError(f"{source} has cosine similarity outside the -1–1 range.")
        by_tier[tier] = parsed
    missing = [tier for tier in TIERS if tier not in by_tier]
    if missing:
        raise ValueError(f"{source} is missing benchmark tiers: {', '.join(missing)}.")
    return [by_tier[tier] for tier in TIERS]


def _validate_alpha(rows: list[dict[str, str]], source: str) -> list[dict[str, float]]:
    """Validate and sort alpha-sweep records."""
    parsed = [
        {
            "alpha": _number(row, "alpha", source),
            "jsr_pct": _number(row, "jsr_pct", source),
            "over_refusal_pct": _number(row, "over_refusal_pct", source),
        }
        for row in rows
    ]
    for row in parsed:
        if row["alpha"] < 0 or not 0 <= row["jsr_pct"] <= 100 or not 0 <= row["over_refusal_pct"] <= 100:
            raise ValueError(f"{source} contains an alpha or rate outside its valid range.")
    if not parsed:
        raise ValueError(f"{source} contains no alpha-sweep rows.")
    parsed.sort(key=lambda row: row["alpha"])
    expected_alphas = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    if tuple(row["alpha"] for row in parsed) != expected_alphas:
        raise ValueError(f"{source} must contain alpha values: {', '.join(map(str, expected_alphas))}.")
    return parsed


def _validate_multiturn(rows: list[dict[str, str]], source: str) -> list[dict[str, float]]:
    """Validate and order three multi-turn drift records."""
    parsed = [
        {
            "turn": _number(row, "turn", source),
            "jsr_pct": _number(row, "jsr_pct", source),
            "mean_cosine_sim": _number(row, "mean_cosine_sim", source),
        }
        for row in rows
    ]
    if len(parsed) != 3:
        raise ValueError(f"{source} must contain exactly three turn records.")
    for row in parsed:
        if row["turn"] not in (1, 2, 3) or not 0 <= row["jsr_pct"] <= 100:
            raise ValueError(f"{source} has an invalid turn or JSR value.")
        if not -1 <= row["mean_cosine_sim"] <= 1:
            raise ValueError(f"{source} has cosine similarity outside the -1–1 range.")
    if {int(row["turn"]) for row in parsed} != {1, 2, 3}:
        raise ValueError(f"{source} must contain exactly turns 1, 2, and 3.")
    return sorted(parsed, key=lambda row: row["turn"])


def _preview_metrics() -> list[dict[str, Any]]:
    """Build visibly illustrative layout data used only when --demo is requested."""
    examples = (
        (1.2, 0.18, 50, 45, 5),
        (8.4, 0.62, 55, 40, 10),
        (1.9, 0.31, 60, 45, 8),
        (1.8, 0.28, 58, 42, 7),
    )
    return [
        {
            "tier": tier,
            "token_fertility": example[0],
            "mean_cosine_sim": example[1],
            "vanilla_jsr": example[2],
            "indicshield_jsr": example[3],
            "over_refusal_rate": example[4],
        }
        for tier, example in zip(TIERS, examples)
    ]


def _preview_alpha() -> list[dict[str, float]]:
    """Build illustrative, non-measured alpha data for preview charts."""
    return [
        {"alpha": alpha, "jsr_pct": 62 + index, "over_refusal_pct": 2 + index * 0.6}
        for index, alpha in enumerate((0.5, 1.0, 1.5, 2.0, 2.5, 3.0))
    ]


def _preview_multiturn() -> list[dict[str, float]]:
    """Build illustrative, non-measured multi-turn data for preview charts."""
    return [
        {"turn": 1.0, "jsr_pct": 60.0, "mean_cosine_sim": 0.31},
        {"turn": 2.0, "jsr_pct": 66.0, "mean_cosine_sim": 0.27},
        {"turn": 3.0, "jsr_pct": 72.0, "mean_cosine_sim": 0.23},
    ]


def _style_axis(axis: Any) -> None:
    """Apply consistent dark panel styling to a Matplotlib axis."""
    axis.set_facecolor(COLORS["panel"])
    axis.tick_params(colors=COLORS["text"])
    axis.xaxis.label.set_color(COLORS["text"])
    axis.yaxis.label.set_color(COLORS["text"])
    axis.title.set_color(COLORS["text"])
    for spine in axis.spines.values():
        spine.set_color(COLORS["grid"])
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _base_figure(title: str, source_label: str, preview: bool, figsize: tuple[float, float]) -> Any:
    """Create a consistently styled figure with a distinct source/preview label."""
    figure = plt.figure(figsize=figsize, facecolor=COLORS["background"])
    figure.suptitle(title, color=COLORS["text"], fontsize=17, fontweight="bold", y=0.98)
    label = (
        f"ILLUSTRATIVE PREVIEW — NOT FINAL OR MEASURED  |  {source_label}"
        if preview
        else f"CSV INPUT — provenance not independently verified  |  {source_label}"
    )
    figure.text(
        0.5,
        0.925,
        label,
        ha="center",
        color=COLORS["amber"] if preview else COLORS["muted"],
        fontsize=9,
    )
    return figure


def _save(figure: Any, output_path: Path) -> None:
    """Save a high-resolution chart and release its figure resources."""
    figure.savefig(
        output_path,
        dpi=300,
        facecolor=COLORS["background"],
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)


def _chart_fertility(rows: list[dict[str, Any]], output_dir: Path, preview: bool) -> None:
    """Plot token fertility and refusal-vector similarity across language tiers."""
    figure = _base_figure(
        "Average Token Fertility and Refusal Similarity",
        "metrics.csv",
        preview,
        (12, 6.5),
    )
    axis = figure.add_subplot(111)
    _style_axis(axis)
    positions = np.arange(len(TIERS))
    fertility_bars = axis.bar(
        positions,
        [row["token_fertility"] for row in rows],
        color=COLORS["cyan"],
        width=0.58,
        label="Avg. Token Fertility",
    )
    axis.set_ylabel("Tokens per whitespace-delimited word")
    axis.set_xticks(positions, TIER_LABELS)
    axis.tick_params(axis="x", labelsize=9)
    right_axis = axis.twinx()
    right_axis.set_facecolor("none")
    right_axis.tick_params(colors=COLORS["text"])
    right_axis.spines["right"].set_color(COLORS["grid"])
    similarity_line = right_axis.plot(
        positions,
        [row["mean_cosine_sim"] for row in rows],
        color=COLORS["amber"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Refusal Vector Cosine Similarity",
    )[0]
    right_axis.set_ylabel("Cosine similarity", color=COLORS["text"])
    right_axis.set_ylim(-1.05, 1.05)
    axis.legend(
        [fertility_bars, similarity_line],
        ["Avg. Token Fertility", "Refusal Vector Cosine Similarity"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=False,
        labelcolor=COLORS["text"],
    )
    figure.subplots_adjust(top=0.79, bottom=0.2, left=0.1, right=0.9)
    _save(figure, output_dir / "chart1_fertility.png")


def _chart_benchmark(rows: list[dict[str, Any]], output_dir: Path, preview: bool) -> None:
    """Plot baseline, shielded JSR, and benign over-refusal by tier."""
    figure = _base_figure("Refusal Benchmark by Language Tier", "metrics.csv", preview, (12, 6.5))
    axis = figure.add_subplot(111)
    _style_axis(axis)
    positions = np.arange(len(TIERS))
    width = 0.24
    series = (
        ("vanilla_jsr", "Vanilla Baseline JSR", COLORS["cyan"], -width),
        ("indicshield_jsr", "IndicShield-RepE JSR", COLORS["emerald"], 0),
        ("over_refusal_rate", "Over-Refusal Rate", COLORS["coral"], width),
    )
    for field, label, color, offset in series:
        bars = axis.bar(
            positions + offset,
            [row[field] for row in rows],
            width,
            color=color,
            label=label,
        )
        axis.bar_label(
            bars,
            labels=[f"{bar.get_height():.1f}%" for bar in bars],
            padding=3,
            fontsize=8,
            color=COLORS["text"],
        )
    axis.set_ylabel("Rate (%)")
    axis.set_ylim(0, 112)
    axis.set_xticks(positions, TIER_LABELS)
    axis.tick_params(axis="x", labelsize=9)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=3,
        frameon=False,
        labelcolor=COLORS["text"],
    )
    figure.subplots_adjust(top=0.79, bottom=0.2, left=0.1, right=0.97)
    _save(figure, output_dir / "chart2_benchmark.png")


def _chart_failure(
    multiturn: list[dict[str, float]],
    alpha: list[dict[str, float]],
    output_dir: Path,
    preview_a: bool,
    preview_b: bool,
) -> None:
    """Plot multi-turn drift and the alpha safety/helpfulness trade-off."""
    figure = plt.figure(figsize=(14, 6.5), facecolor=COLORS["background"])
    figure.suptitle(
        "Failure Analysis: Multi-Turn Drift and Alpha Trade-Off",
        color=COLORS["text"],
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    panels = (
        "Panel A — Multi-Turn Gradual Drift",
        "Panel B — Alpha Safety–Helpfulness Trade-Off",
    )
    for index, (title, preview) in enumerate(zip(panels, (preview_a, preview_b))):
        label = "PREVIEW · NOT MEASURED" if preview else "CSV · provenance not verified"
        figure.text(
            0.26 if index == 0 else 0.74,
            0.91,
            f"{title}\n{label}",
            ha="center",
            va="top",
            color=COLORS["amber"] if preview else COLORS["text"],
            fontsize=11,
            fontweight="bold",
        )

    left = figure.add_subplot(121)
    right = figure.add_subplot(122)
    _style_axis(left)
    _style_axis(right)

    turn_labels = {
        1: "Turn 1\nDirect unsafe",
        2: "Turn 2\n1 benign + unsafe",
        3: "Turn 3\n2 benign + unsafe",
    }
    positions = np.arange(3)
    bars = left.bar(
        positions,
        [row["jsr_pct"] for row in multiturn],
        width=0.52,
        color=COLORS["coral"],
        label="Jailbreak Success Rate",
    )
    left.bar_label(
        bars,
        labels=[f"{bar.get_height():.1f}%" for bar in bars],
        padding=3,
        fontsize=8,
        color=COLORS["text"],
    )
    left.set_xticks(positions, [turn_labels[int(row["turn"])] for row in multiturn], fontsize=8)
    left.set_ylabel("Jailbreak Success Rate (%)")
    left.set_ylim(0, 112)
    left_right = left.twinx()
    cosine_line = left_right.plot(
        positions,
        [row["mean_cosine_sim"] for row in multiturn],
        color=COLORS["amber"],
        marker="o",
        linewidth=2.3,
        label="Mean Cosine Similarity",
    )[0]
    left_right.set_ylim(-1.05, 1.05)
    left_right.set_ylabel("Mean cosine similarity", color=COLORS["text"])
    left_right.tick_params(colors=COLORS["text"])
    left_right.spines["right"].set_color(COLORS["grid"])
    left.legend(
        [bars, cosine_line],
        ["Jailbreak Success Rate", "Mean Cosine Similarity"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=1,
        frameon=False,
        labelcolor=COLORS["text"],
        fontsize=8,
    )

    right.plot(
        [row["alpha"] for row in alpha],
        [row["jsr_pct"] for row in alpha],
        color=COLORS["coral"],
        marker="o",
        linewidth=2.4,
        label="Jailbreak Success Rate",
    )
    right.plot(
        [row["alpha"] for row in alpha],
        [row["over_refusal_pct"] for row in alpha],
        color=COLORS["emerald"],
        marker="s",
        linewidth=2.4,
        label="Over-Refusal Rate",
    )
    right.set_xlabel("Intervention strength (alpha)")
    right.set_ylabel("Rate (%)")
    right.set_ylim(bottom=0)
    right.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=1,
        frameon=False,
        labelcolor=COLORS["text"],
        fontsize=8,
    )
    figure.subplots_adjust(top=0.72, bottom=0.22, left=0.08, right=0.92, wspace=0.46)
    _save(figure, output_dir / "chart3_failure.png")


def generate_charts(results_dir: Path = RESULTS_DIR, demo: bool = False) -> None:
    """Load available benchmark CSVs and export the three requested PNGs."""
    metrics_source = _read_csv(results_dir / "metrics.csv", METRICS_COLUMNS)
    metrics_preview = metrics_source is None
    if metrics_preview and not demo:
        raise ValueError(
            f"{results_dir / 'metrics.csv'} is missing or empty. Run the benchmark first, "
            "or pass --demo to generate explicitly non-final preview charts."
        )
    metrics = (
        _preview_metrics()
        if metrics_preview
        else _validate_metrics(metrics_source, "metrics.csv")
    )

    alpha_source = _read_csv(results_dir / "alpha_sweep.csv", ALPHA_COLUMNS)
    alpha_preview = alpha_source is None
    if alpha_preview and not demo:
        raise ValueError(
            f"{results_dir / 'alpha_sweep.csv'} is missing or empty; "
            "provide real sweep rows or pass --demo for an illustrative preview."
        )
    alpha = _preview_alpha() if alpha_preview else _validate_alpha(alpha_source, "alpha_sweep.csv")

    multiturn_source = _read_csv(results_dir / "multiturn_drift.csv", MULTITURN_COLUMNS)
    multiturn_preview = multiturn_source is None
    if multiturn_preview and not demo:
        raise ValueError(
            f"{results_dir / 'multiturn_drift.csv'} is missing or empty; "
            "provide real drift rows or pass --demo for an illustrative preview."
        )
    multiturn = (
        _preview_multiturn()
        if multiturn_preview
        else _validate_multiturn(multiturn_source, "multiturn_drift.csv")
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    _chart_fertility(metrics, results_dir, metrics_preview)
    _chart_benchmark(metrics, results_dir, metrics_preview)
    _chart_failure(multiturn, alpha, results_dir, multiturn_preview, alpha_preview)
    print(f"Generated three 300-DPI charts in {results_dir}.")
    if metrics_preview or alpha_preview or multiturn_preview:
        print("At least one chart uses illustrative preview data; it is not a benchmark result.")


def main() -> None:
    """Parse chart-generation arguments and report validation failures clearly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing benchmark CSVs and receiving PNG charts.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Allow clearly watermarked illustrative data when an input CSV is missing or empty.",
    )
    args = parser.parse_args()
    try:
        generate_charts(args.results_dir, args.demo)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
