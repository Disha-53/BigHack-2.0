"""Generate dark, presentation-ready IndicShield benchmark charts."""

import argparse
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
TIERS = [
    "english",
    "native_kannada",
    "kanglish_hinglish",
    "codemixed_genz_slang",
]
TIER_LABELS = [
    "English",
    "Native\nKannada",
    "Kanglish /\nHinglish",
    "Code-Mixed +\nGen-Z Slang",
]
COLORS = {
    "background": "#0F172A",
    "panel": "#1E293B",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "grid": "#334155",
    "cyan": "#38BDF8",
    "emerald": "#10B981",
    "coral": "#F43F5E",
    "amber": "#F59E0B",
}
DEMO_STATUS = "illustrative_demo"


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().casefold()).strip("_")


def _find_column(frame: pd.DataFrame, *aliases: str) -> str | None:
    columns = {_normalize_column(str(column)): str(column) for column in frame.columns}
    for alias in aliases:
        found = columns.get(_normalize_column(alias))
        if found is not None:
            return found
    return None


def _tier_key(value: object) -> str:
    normalized = _normalize_column(str(value))
    aliases = {
        "english": "english",
        "native_kannada": "native_kannada",
        "kannada": "native_kannada",
        "kanglish": "kanglish_hinglish",
        "hinglish": "kanglish_hinglish",
        "kanglish_hinglish": "kanglish_hinglish",
        "codemixed_genz_slang": "codemixed_genz_slang",
        "code_mixed_genz_slang": "codemixed_genz_slang",
        "code_mixed_gen_z_slang": "codemixed_genz_slang",
    }
    return aliases.get(normalized, normalized)


def _number_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{label} contains missing or non-numeric values.")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite values.")
    return values.astype(float)


def _as_percent(values: pd.Series) -> pd.Series:
    if not values.empty and values.abs().max() <= 1:
        return values * 100
    return values


def _load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty.")

    tier_col = _find_column(frame, "tier", "linguistic_tier", "dataset")
    fertility_col = _find_column(
        frame, "token_fertility", "mean_token_fertility", "fertility"
    )
    cosine_col = _find_column(
        frame, "mean_cosine_sim", "mean_cosine_similarity",
        "refusal_vector_cosine", "cosine_similarity", "cosine",
    )
    vanilla_col = _find_column(frame, "vanilla_jsr", "vanilla_jsr_pct")
    shield_col = _find_column(
        frame, "indicshield_jsr", "indicshield_jsr_pct", "shield_jsr"
    )
    over_col = _find_column(
        frame, "over_refusal_rate", "over_refusal_pct", "over_refusal"
    )
    missing = [
        name
        for name, column in (
            ("tier", tier_col),
            ("token_fertility", fertility_col),
            ("mean_cosine_sim", cosine_col),
            ("vanilla_jsr", vanilla_col),
            ("indicshield_jsr", shield_col),
            ("over_refusal_rate", over_col),
        )
        if column is None
    ]

    # Also accept long-form rows emitted by earlier benchmark integrations.
    if missing:
        system_col = _find_column(frame, "system", "model", "method")
        jsr_col = _find_column(frame, "jsr", "jsr_pct", "jailbreak_success_rate")
        refusal_col = _find_column(frame, "refusal_rate", "refusal_rate_pct")
        if tier_col and system_col and (jsr_col or refusal_col):
            jsr_column = jsr_col or refusal_col
            rows = frame.copy()
            status_col = _find_column(rows, "status")
            if status_col:
                status = rows[status_col].astype(str).str.casefold()
                rows = rows.loc[~status.isin({
                    "not_implemented", "pending", "placeholder", DEMO_STATUS,
                })]
            rows["_tier"] = rows[tier_col].map(_tier_key)
            rows["_system"] = rows[system_col].astype(str).str.casefold()
            rows["_rate"] = _number_series(rows, jsr_column, jsr_column)
            if refusal_col and not jsr_col:
                rows["_rate"] = 1 - rows["_rate"] if rows["_rate"].max() <= 1 else 100 - rows["_rate"]
            rows["_rate"] = _as_percent(rows["_rate"])
            rows["_fertility"] = (
                _number_series(rows, fertility_col, fertility_col)
                if fertility_col else np.nan
            )
            rows["_cosine"] = (
                _number_series(rows, cosine_col, cosine_col)
                if cosine_col else np.nan
            )
            rows["_over"] = (
                _as_percent(_number_series(rows, over_col, over_col))
                if over_col else np.nan
            )
            pivot_rows = []
            for tier in TIERS:
                group = rows.loc[rows["_tier"] == tier]
                baseline = group.loc[group["_system"].str.contains("vanilla|baseline")]
                shield = group.loc[group["_system"].str.contains("indicshield|shield|rep[eé]")]
                if baseline.empty or shield.empty:
                    continue
                pivot_rows.append({
                    "tier": tier,
                    "token_fertility": group["_fertility"].mean(),
                    "mean_cosine_sim": group["_cosine"].mean(),
                    "vanilla_jsr": baseline["_rate"].mean(),
                    "indicshield_jsr": shield["_rate"].mean(),
                    "over_refusal_rate": group["_over"].mean(),
                })
            if len(pivot_rows) == len(TIERS):
                return pd.DataFrame(pivot_rows)
        raise ValueError(
            f"{path} is missing chart fields: {', '.join(missing)}. "
            "Expected tier, token_fertility, mean_cosine_sim, vanilla_jsr, "
            "indicshield_jsr, and over_refusal_rate columns."
        )

    result = pd.DataFrame({
        "tier": frame[tier_col].map(_tier_key),
        "token_fertility": _number_series(frame, fertility_col, fertility_col),
        "mean_cosine_sim": _number_series(frame, cosine_col, cosine_col),
        "vanilla_jsr": _as_percent(_number_series(frame, vanilla_col, vanilla_col)),
        "indicshield_jsr": _as_percent(_number_series(frame, shield_col, shield_col)),
        "over_refusal_rate": _as_percent(_number_series(frame, over_col, over_col)),
    })
    return _ordered_tiers(result, path)


def _ordered_tiers(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if frame["tier"].duplicated().any():
        frame = frame.groupby("tier", as_index=False).mean(numeric_only=True)
    available = set(frame["tier"])
    missing = [tier for tier in TIERS if tier not in available]
    if missing:
        raise ValueError(f"{path} has no measured rows for tiers: {', '.join(missing)}.")
    return frame.set_index("tier").loc[TIERS].reset_index()


def _load_drift(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    turn_col = _find_column(frame, "turn", "turn_label", "scenario")
    jsr_col = _find_column(frame, "jsr_pct", "jsr", "attack_success_rate")
    cosine_col = _find_column(
        frame, "mean_cosine_sim", "mean_cosine_similarity",
        "cosine_similarity", "cosine",
    )
    if frame.empty or not all((turn_col, jsr_col, cosine_col)):
        raise ValueError(
            f"{path} must contain turn, jsr_pct, and mean_cosine_sim columns."
        )
    result = pd.DataFrame({
        "turn": frame[turn_col].astype(str),
        "jsr_pct": _as_percent(_number_series(frame, jsr_col, jsr_col)),
        "mean_cosine_sim": _number_series(frame, cosine_col, cosine_col),
    })
    status_col = _find_column(frame, "status")
    if status_col:
        status = frame[status_col].astype(str).str.casefold()
        result = result.loc[~status.isin({
            "not_implemented", "pending", "placeholder",
        })].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"{path} has no measured drift rows.")
    return result


def _load_alpha(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    alpha_col = _find_column(frame, "alpha", "steering_strength")
    jsr_col = _find_column(frame, "jsr_pct", "jsr", "jailbreak_success_rate")
    over_col = _find_column(
        frame, "over_refusal_pct", "over_refusal_rate", "over_refusal"
    )
    if frame.empty or not all((alpha_col, jsr_col, over_col)):
        raise ValueError(
            f"{path} must contain alpha, jsr_pct, and over_refusal_pct columns."
        )
    result = pd.DataFrame({
        "alpha": _number_series(frame, alpha_col, alpha_col),
        "jsr_pct": _as_percent(_number_series(frame, jsr_col, jsr_col)),
        "over_refusal_pct": _as_percent(_number_series(frame, over_col, over_col)),
    })
    status_col = _find_column(frame, "status")
    if status_col:
        status = frame[status_col].astype(str).str.casefold()
        result = result.loc[~status.isin({
            "not_implemented", "pending", "placeholder",
        })].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"{path} has no measured alpha-sweep rows.")
    return result.sort_values("alpha").reset_index(drop=True)


def _demo_frames() -> dict[str, pd.DataFrame]:
    return {
        "metrics.csv": pd.DataFrame({
            "tier": TIERS,
            "token_fertility": [1.18, 2.46, 2.12, 1.91],
            "mean_cosine_sim": [0.79, 0.68, 0.84, 0.70],
            "vanilla_jsr": [46.0, 61.0, 68.0, 55.0],
            "indicshield_jsr": [17.0, 28.0, 22.0, 31.0],
            "over_refusal_rate": [3.0, 5.0, 4.0, 6.0],
            "status": [DEMO_STATUS] * len(TIERS),
        }),
        "multiturn_drift.csv": pd.DataFrame({
            "turn": [
                "Turn 1 · Direct",
                "Turn 2 · +1 Benign Turn",
                "Turn 3 · +2 Benign Turns",
            ],
            "jsr_pct": [12.0, 36.0, 64.0],
            "mean_cosine_sim": [0.82, 0.70, 0.57],
            "status": [DEMO_STATUS] * 3,
        }),
        "alpha_sweep.csv": pd.DataFrame({
            "alpha": [0.5, 1.0, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0],
            "jsr_pct": [54.0, 40.0, 27.0, 24.0, 21.0, 17.0, 12.0, 9.0],
            "over_refusal_pct": [1.5, 2.0, 3.2, 3.7, 4.5, 6.5, 11.0, 17.0],
            "status": [DEMO_STATUS] * 8,
        }),
    }


def _only_demo_or_missing(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return False
    status_col = _find_column(frame, "status")
    if frame.empty:
        return True
    if status_col is None:
        return False
    statuses = set(frame[status_col].astype(str).str.casefold())
    return statuses <= {"not_implemented", "pending", "placeholder", DEMO_STATUS}


def _prepare_inputs(use_demo: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if use_demo:
        for filename, frame in _demo_frames().items():
            path = RESULTS_DIR / filename
            if _only_demo_or_missing(path):
                frame.to_csv(path, index=False)
                print(f"Wrote illustrative demo data: {path}")

    paths = {
        name: RESULTS_DIR / name
        for name in ("metrics.csv", "multiturn_drift.csv", "alpha_sweep.csv")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing benchmark CSV(s): " + ", ".join(missing)
            + ". Run with --demo for illustrative charts."
        )
    return (
        _load_metrics(paths["metrics.csv"]),
        _load_drift(paths["multiturn_drift.csv"]),
        _load_alpha(paths["alpha_sweep.csv"]),
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["panel"])
    axis.tick_params(colors=COLORS["muted"], labelsize=10, pad=8)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(COLORS["grid"])


def _style_figure(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.patch.set_facecolor(COLORS["background"])
    for axis in axes:
        _style_axis(axis)


def _style_legend(legend: object) -> None:
    frame = legend.get_frame()
    frame.set_facecolor(COLORS["panel"])
    frame.set_edgecolor(COLORS["grid"])
    for text in legend.get_texts():
        text.set_color(COLORS["text"])


def chart_fertility(metrics: pd.DataFrame) -> Path:
    fig, axis_left = plt.subplots(figsize=(12.8, 7.2), constrained_layout=False)
    axis_right = axis_left.twinx()
    _style_figure(fig, [axis_left, axis_right])
    axis_right.patch.set_visible(False)
    axis_right.grid(False)
    x = np.arange(len(TIERS))
    bars = axis_left.bar(
        x,
        metrics["token_fertility"],
        width=0.58,
        color=COLORS["cyan"],
        edgecolor="#7DD3FC",
        linewidth=0.8,
        label="Token Fertility",
        zorder=3,
    )
    line = axis_right.plot(
        x,
        metrics["mean_cosine_sim"],
        color=COLORS["amber"],
        marker="o",
        markersize=9,
        linewidth=2.8,
        label="Mean Cosine Similarity",
        zorder=4,
    )[0]
    axis_left.set_xticks(x, TIER_LABELS)
    axis_left.set_ylabel("Token fertility · tokens / whitespace word", color=COLORS["muted"], labelpad=12)
    axis_right.set_ylabel("Mean cosine similarity", color=COLORS["muted"], labelpad=12)
    axis_left.set_ylim(0, max(metrics["token_fertility"]) * 1.28)
    axis_right.set_ylim(0, 1)
    axis_left.set_title(
        "Token Fertility vs. Refusal Signal",
        color=COLORS["text"],
        fontsize=19,
        fontweight="bold",
        loc="left",
        pad=22,
    )
    axis_left.bar_label(bars, fmt="%.2fx", padding=6, color=COLORS["text"], fontsize=10)
    legend = axis_left.legend(
        [bars, line], ["Token Fertility", "Mean Cosine Similarity"],
        loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2,
        frameon=True, fontsize=10,
    )
    _style_legend(legend)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.20, top=0.79)
    path = RESULTS_DIR / "chart1_fertility.png"
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def chart_benchmark(metrics: pd.DataFrame) -> Path:
    fig, axis = plt.subplots(figsize=(12.8, 7.2), constrained_layout=False)
    _style_figure(fig, [axis])
    x = np.arange(len(TIERS))
    width = 0.23
    series = (
        ("vanilla_jsr", "Vanilla Baseline JSR", COLORS["coral"]),
        ("indicshield_jsr", "IndicShield-RepE JSR", COLORS["emerald"]),
        ("over_refusal_rate", "Over-Refusal Rate", COLORS["amber"]),
    )
    for index, (column, label, color) in enumerate(series):
        values = metrics[column].to_numpy(dtype=float)
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            color=color,
            edgecolor=COLORS["background"],
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:g}%" for value in values],
            padding=5,
            color=COLORS["text"],
            fontsize=9,
        )
    axis.set_xticks(x, TIER_LABELS)
    axis.set_ylabel("Rate (%)", color=COLORS["muted"], labelpad=12)
    axis.set_ylim(0, max(100.0, float(metrics[list(c for c, _, _ in series)].to_numpy().max()) * 1.22))
    axis.set_title(
        "Refusal & Helpfulness by Linguistic Tier",
        color=COLORS["text"],
        fontsize=19,
        fontweight="bold",
        loc="left",
        pad=22,
    )
    legend = axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3,
        frameon=True, fontsize=10,
    )
    _style_legend(legend)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.20, top=0.79)
    path = RESULTS_DIR / "chart2_benchmark.png"
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def chart_failure(drift: pd.DataFrame, alpha: pd.DataFrame) -> Path:
    fig, (axis_drift, axis_alpha) = plt.subplots(
        1, 2, figsize=(16, 7.5), constrained_layout=False
    )
    _style_figure(fig, [axis_drift, axis_alpha])

    turns = np.arange(len(drift))
    drift_right = axis_drift.twinx()
    _style_axis(drift_right)
    drift_right.patch.set_visible(False)
    drift_right.grid(False)
    axis_drift.fill_between(
        turns,
        drift["jsr_pct"].to_numpy(dtype=float),
        color=COLORS["coral"],
        alpha=0.12,
    )
    drift_jsr = axis_drift.plot(
        turns,
        drift["jsr_pct"],
        color=COLORS["coral"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="JSR (%)",
    )[0]
    drift_cos = drift_right.plot(
        turns,
        drift["mean_cosine_sim"],
        color=COLORS["cyan"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Mean Cosine Similarity",
    )[0]
    axis_drift.set_xticks(turns, drift["turn"], rotation=0, ha="center")
    axis_drift.set_ylabel("Jailbreak Success Rate (%)", color=COLORS["muted"])
    drift_right.set_ylabel("Mean cosine similarity", color=COLORS["muted"])
    axis_drift.set_ylim(0, max(100.0, float(drift["jsr_pct"].max()) * 1.2))
    drift_right.set_ylim(0, 1)
    axis_drift.set_title(
        "A · Multi-Turn Gradual Drift",
        color=COLORS["text"],
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    drift_legend = axis_drift.legend(
        [drift_jsr, drift_cos], [drift_jsr.get_label(), drift_cos.get_label()],
        loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2,
        frameon=True, fontsize=9,
    )
    _style_legend(drift_legend)

    axis_alpha.axvspan(
        1.5, 1.8, color=COLORS["emerald"], alpha=0.14,
        label="Optimal zone · α 1.5–1.8",
    )
    alpha_jsr = axis_alpha.plot(
        alpha["alpha"], alpha["jsr_pct"],
        color=COLORS["coral"], marker="o", linewidth=2.5,
        markersize=7, label="JSR (%)",
    )[0]
    alpha_over = axis_alpha.plot(
        alpha["alpha"], alpha["over_refusal_pct"],
        color=COLORS["amber"], marker="o", linewidth=2.5,
        markersize=7, label="Over-Refusal (%)",
    )[0]
    axis_alpha.set_xlabel("Steering strength · α", color=COLORS["muted"], labelpad=10)
    axis_alpha.set_ylabel("Rate (%)", color=COLORS["muted"])
    axis_alpha.set_ylim(0, 100)
    axis_alpha.set_xlim(0.4, 3.1)
    axis_alpha.set_xticks(np.arange(0.5, 3.01, 0.5))
    axis_alpha.set_title(
        "B · Alpha Safety–Helpfulness Trade-Off",
        color=COLORS["text"],
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=16,
    )
    alpha_legend = axis_alpha.legend(
        [alpha_jsr, alpha_over, axis_alpha.patches[0]],
        [alpha_jsr.get_label(), alpha_over.get_label(), "Optimal zone · α 1.5–1.8"],
        loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
        frameon=True, fontsize=8,
    )
    _style_legend(alpha_legend)
    fig.suptitle(
        "Failure Analysis & Steering Calibration",
        color=COLORS["text"],
        fontsize=19,
        fontweight="bold",
        x=0.04,
        ha="left",
        y=0.99,
    )
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.19, top=0.77, wspace=0.38)
    path = RESULTS_DIR / "chart3_failure.png"
    fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Create clearly labeled illustrative CSVs when inputs are missing or "
            "contain only the benchmark scaffold."
        ),
    )
    args = parser.parse_args()
    try:
        metrics, drift, alpha = _prepare_inputs(args.demo)
        generated = (
            chart_fertility(metrics),
            chart_benchmark(metrics),
            chart_failure(drift, alpha),
        )
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        print(f"Chart generation failed: {error}", file=sys.stderr)
        return 1

    for path in generated:
        print(f"Saved 300-DPI chart: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
