"""Safe Streamlit inspector for local IndicShield CSVs and optional live inference."""

import csv
import importlib.util
from pathlib import Path
from typing import Any

import streamlit as st

from src.evaluator import is_refusal, redact_if_unsafe

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
REDACTED_RESPONSE = "[REDACTED - UNSAFE RESPONSE]"
METRICS_COLUMNS = (
    "tier",
    "token_fertility",
    "mean_cosine_sim",
    "vanilla_jsr",
    "indicshield_jsr",
    "over_refusal_rate",
    "avg_latency_ms",
)
EVAL_LOG_COLUMNS = (
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
TIERS = ("english", "native_kannada", "kanglish_hinglish", "codemixed_genz_slang")

st.set_page_config(
    page_title="IndicShield-RepE Research Inspector",
    page_icon="🛡️",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #0F172A; color: #F8FAFC; }
    [data-testid="stMetric"], [data-testid="stExpander"] {
        background: #1E293B; border: 1px solid #334155; border-radius: 0.75rem;
    }
    .mode-card {
        background: #1E293B; border: 1px solid #334155; border-radius: 0.75rem;
        padding: 1rem; min-height: 7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _cached_file_exists(repo_id: str, filename: str) -> bool:
    """Check the local Hugging Face cache without attempting a network download."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cached = try_to_load_from_cache(repo_id, filename)
    return isinstance(cached, str) and Path(cached).is_file()


def _live_prerequisites() -> tuple[bool, list[str]]:
    """Return whether local checkpoint, hook, GPU, and model weights are available."""
    missing: list[str] = []
    if not (RESULTS_DIR / "refusal_vec.pt").is_file():
        missing.append("results/refusal_vec.pt is missing")
    if importlib.util.find_spec("src.indicshield_hook") is None:
        missing.append("src.indicshield_hook is unavailable")

    try:
        import torch
    except ImportError:
        missing.append("PyTorch is not installed")
        return False, missing

    if not torch.cuda.is_available():
        missing.append("no CUDA-capable GPU is available")

    cached_config = _cached_file_exists(MODEL_NAME, "config.json")
    cached_weights = any(
        _cached_file_exists(MODEL_NAME, filename)
        for filename in (
            "model.safetensors.index.json",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "pytorch_model.bin",
        )
    )
    if not cached_config or not cached_weights:
        missing.append(f"{MODEL_NAME} weights are not present in the local model cache")
    return not missing, missing


@st.cache_resource(show_spinner="Loading local Qwen weights...")
def _load_model() -> tuple[Any, Any]:
    """Load cached model weights in float16 without downloading missing files."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


def _load_refusal_vector() -> tuple[int, Any]:
    """Read the selected layer and matching refusal vector from the local checkpoint."""
    import torch

    checkpoint_path = RESULTS_DIR / "refusal_vec.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("The refusal-vector checkpoint must contain a dictionary.")
    best_layer = checkpoint.get("best_layer")
    vectors = checkpoint.get("vectors")
    if not isinstance(best_layer, int) or not isinstance(vectors, dict):
        raise ValueError("The checkpoint must contain integer best_layer and layer vectors.")
    refusal_vec = vectors.get(best_layer, vectors.get(str(best_layer)))
    if not isinstance(refusal_vec, torch.Tensor) or refusal_vec.ndim != 1:
        raise ValueError(f"The checkpoint has no valid vector for layer {best_layer}.")
    return best_layer, refusal_vec


def _generate(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    """Generate a deterministic response using the model's chat template."""
    import torch

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
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


class _CosineProbe:
    """Observe last-token prompt similarity before the steering hook modifies it."""

    def __init__(self, refusal_vec: Any) -> None:
        self.refusal_vec = refusal_vec.detach()
        self.values: list[float] = []

    def __call__(self, module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        """Record prefill cosine similarity for display with the live result."""
        import torch

        del module, inputs
        activations = output if isinstance(output, torch.Tensor) else output[0]
        if not isinstance(activations, torch.Tensor) or activations.ndim != 3:
            return
        vector = self.refusal_vec.to(device=activations.device, dtype=torch.float32)
        if activations.shape[-1] != vector.shape[0]:
            raise ValueError("Refusal vector and model hidden size do not match.")
        similarity = torch.nn.functional.cosine_similarity(
            activations[:, -1, :].float(),
            vector.expand(activations.shape[0], -1),
            dim=-1,
        )
        self.values.append(float(similarity.mean().item()))


def _run_live(prompt: str, tau: float, alpha: float, is_harmful: bool) -> dict[str, Any]:
    """Run both generations and return only refusal-safe, displayable response text."""
    from src.indicshield_hook import IndicShieldHook

    model, tokenizer = _load_model()
    best_layer, refusal_vec = _load_refusal_vector()
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or best_layer < 0 or best_layer >= len(layers):
        raise ValueError(f"The model does not expose decoder layer {best_layer}.")
    layer = layers[best_layer]
    messages = [{"role": "user", "content": prompt}]

    vanilla_response = _generate(model, tokenizer, messages, max_new_tokens=64)
    probe = _CosineProbe(refusal_vec)
    probe_handle = layer.register_forward_hook(probe)
    hook = IndicShieldHook(refusal_vec, best_layer, tau=tau, alpha=alpha)
    hook.attach(layer)
    try:
        shield_response = _generate(model, tokenizer, messages, max_new_tokens=64)
    finally:
        hook.remove()
        probe_handle.remove()

    vanilla_refused = is_refusal(vanilla_response)
    shield_refused = is_refusal(shield_response)
    hook_stats = hook.get_stats()
    return {
        "vanilla_display": redact_if_unsafe(vanilla_response, is_harmful, vanilla_refused),
        "indicshield_display": redact_if_unsafe(shield_response, is_harmful, shield_refused),
        "vanilla_refused": vanilla_refused,
        "indicshield_refused": shield_refused,
        "cosine_similarity": probe.values[0] if probe.values else None,
        "latency_ms": sum(hook.latencies_ms) if hook.latencies_ms else None,
        "triggered": hook_stats["triggered"],
    }


def _read_metrics() -> tuple[Any | None, str | None]:
    """Load a usable metrics CSV, reporting missing or malformed input."""
    return _read_result_csv("metrics.csv", METRICS_COLUMNS)


def _read_result_csv(filename: str, expected_columns: tuple[str, ...]) -> tuple[Any | None, str | None]:
    """Read a results CSV only when its header matches the measured schema."""
    path = RESULTS_DIR / filename
    if not path.is_file() or path.stat().st_size == 0:
        return None, None
    try:
        with path.open(encoding="utf-8-sig", newline="") as result_file:
            reader = csv.DictReader(result_file)
            if tuple(reader.fieldnames or ()) != expected_columns:
                return None, f"{path.name} does not match the expected CSV schema."
        import pandas as pd

        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError, ValueError, csv.Error) as error:
        return None, f"Could not read {path.name}: {error}"
    return frame, None


def _parse_bool(value: Any, default: bool = False) -> bool:
    """Parse CSV boolean values without treating non-empty false strings as true."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return default


def _format_percent(value: Any) -> str:
    """Format a stored ratio or percentage-point value for display only."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not (-float("inf") < number < float("inf")):
        return "—"
    percentage = number * 100 if 0 <= number <= 1 else number
    return f"{percentage:.1f}%"


def _format_rate_columns(frame: Any, columns: tuple[str, ...]) -> Any:
    """Return a display copy with rate columns formatted as percentages."""
    display_frame = frame.copy()
    for column in columns:
        if column in display_frame.columns:
            display_frame[column] = display_frame[column].map(_format_percent)
    return display_frame


def _measured_result(row: dict[str, Any], tau: float) -> dict[str, Any]:
    """Prepare a measured log row for safe display, deriving its gate at selected tau."""
    harmful = _parse_bool(row["is_harmful"], default=True)
    vanilla_refused = _parse_bool(row["vanilla_refused"])
    shield_refused = _parse_bool(row["indicshield_refused"])
    cosine_similarity = float(row["cosine_sim"])
    return {
        "mode": "measured",
        "prompt": str(row["prompt"]),
        "is_harmful": harmful,
        "vanilla_display": redact_if_unsafe(
            str(row["vanilla_response"]), harmful, vanilla_refused
        ),
        "indicshield_display": redact_if_unsafe(
            str(row["indicshield_response"]), harmful, shield_refused
        ),
        "vanilla_refused": vanilla_refused,
        "indicshield_refused": shield_refused,
        "cosine_similarity": cosine_similarity,
        "triggered": cosine_similarity >= tau,
        "latency_ms": float(row["latency_ms"]),
        "id": str(row["id"]),
        "tier": str(row["tier"]),
        "tau": tau,
    }


def _sync_measured_selection(rows_by_id: dict[str, dict[str, Any]]) -> None:
    """Copy the selected measured record into the existing prompt controls."""
    case_id = st.session_state["measured_case_id"]
    row = rows_by_id.get(case_id)
    if row is None:
        return
    st.session_state["research_prompt"] = str(row["prompt"])
    st.session_state["research_tier"] = str(row["tier"])
    st.session_state["reviewed_benign"] = not _parse_bool(row["is_harmful"], default=True)


def _percentage_number(value: Any) -> float:
    """Convert a stored ratio or percentage-point value to percentage points."""
    number = float(value)
    return number * 100 if 0 <= number <= 1 else number


def _style_chart(chart: Any) -> Any:
    """Apply the existing dark palette to an interactive Altair chart."""
    import altair as alt

    return (
        chart.configure(
            background="#0F172A",
            axis=alt.AxisConfig(
                labelColor="#CBD5E1",
                titleColor="#F8FAFC",
                gridColor="#334155",
                domainColor="#64748B",
            ),
            legend=alt.LegendConfig(labelColor="#CBD5E1", titleColor="#F8FAFC"),
            title=alt.TitleConfig(color="#F8FAFC"),
        )
        .configure_view(stroke=None)
    )


def _render_interactive_charts(
    eval_frame: Any | None,
    metrics_frame: Any | None,
    alpha_frame: Any | None,
    drift_frame: Any | None,
    tau: float,
) -> None:
    """Render interactive charts exclusively from the supplied measured CSV rows."""
    import altair as alt
    import pandas as pd

    st.subheader("Interactive Benchmark Charts")
    if eval_frame is not None and not eval_frame.empty:
        threshold_data = eval_frame.copy()
        threshold_data["cosine_sim"] = pd.to_numeric(
            threshold_data["cosine_sim"], errors="coerce"
        )
        threshold_data["is_harmful"] = threshold_data["is_harmful"].map(
            lambda value: _parse_bool(value, default=True)
        )
        threshold_data = threshold_data.dropna(subset=["cosine_sim"])
        harmful_rows = threshold_data[threshold_data["is_harmful"]]
        benign_rows = threshold_data[~threshold_data["is_harmful"]]
        caught_count = int((harmful_rows["cosine_sim"] >= tau).sum())
        false_trigger_count = int((benign_rows["cosine_sim"] >= tau).sum())

        st.markdown("#### Interactive Live Threshold (τ) Calibration Curve")
        harmful_kpi, benign_kpi = st.columns(2)
        with harmful_kpi:
            st.metric(
                "Harmful Prompts Caught at τ",
                f"{caught_count} / {len(harmful_rows)}",
                delta=(
                    "—"
                    if harmful_rows.empty
                    else f"{100 * caught_count / len(harmful_rows):.1f}%"
                ),
            )
        with benign_kpi:
            st.metric(
                "Benign Prompts False-Triggered at τ",
                f"{false_trigger_count} / {len(benign_rows)}",
                delta=(
                    "—"
                    if benign_rows.empty
                    else f"{100 * false_trigger_count / len(benign_rows):.1f}%"
                ),
                delta_color="inverse",
            )

        threshold_rows: list[dict[str, Any]] = []
        for step in range(29):
            threshold = round(0.12 + step * 0.005, 3)
            for label, rows in (
                ("Harmful Detection Rate", harmful_rows),
                ("Benign False-Trigger Rate", benign_rows),
            ):
                if rows.empty:
                    continue
                count = int((rows["cosine_sim"] >= threshold).sum())
                threshold_rows.append(
                    {
                        "tau": threshold,
                        "rate_pct": 100 * count / len(rows),
                        "prompt_class": label,
                    }
                )

        if threshold_rows:
            threshold_domain = [min(0.12, tau), max(0.26, tau)]
            threshold_chart = (
                alt.Chart(pd.DataFrame(threshold_rows))
                .mark_line(point=True)
                .encode(
                    x=alt.X(
                        "tau:Q",
                        title="Gate threshold (τ)",
                        scale=alt.Scale(domain=threshold_domain),
                    ),
                    y=alt.Y("rate_pct:Q", title="Prompts triggered (%)"),
                    color=alt.Color(
                        "prompt_class:N",
                        title=None,
                        scale=alt.Scale(
                            domain=[
                                "Harmful Detection Rate",
                                "Benign False-Trigger Rate",
                            ],
                            range=["#38BDF8", "#F43F5E"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("tau:Q", title="τ", format=".3f"),
                        alt.Tooltip("prompt_class:N", title="Rate"),
                        alt.Tooltip("rate_pct:Q", title="Prompts (%)", format=".1f"),
                    ],
                )
            )
            threshold_rule = (
                alt.Chart(pd.DataFrame({"tau": [tau]}))
                .mark_rule(color="#F59E0B", strokeDash=[6, 4], size=2)
                .encode(
                    x=alt.X("tau:Q", scale=alt.Scale(domain=threshold_domain)),
                    tooltip=[alt.Tooltip("tau:Q", title="Selected τ", format=".3f")],
                )
            )
            st.altair_chart(
                _style_chart(
                    alt.layer(threshold_chart, threshold_rule)
                    .properties(height=300)
                    .interactive()
                ),
                width="stretch",
            )
    else:
        st.info("No measured eval_logs.csv rows are available for threshold calibration.")

    if metrics_frame is not None and not metrics_frame.empty:
        st.markdown("#### Token Fertility vs. Mean Cosine Similarity")
        scatter = (
            alt.Chart(metrics_frame)
            .mark_circle(size=180)
            .encode(
                x=alt.X("token_fertility:Q", title="Average token fertility"),
                y=alt.Y("mean_cosine_sim:Q", title="Mean cosine similarity"),
                color=alt.Color("tier:N", title="Language tier", sort=list(TIERS)),
                tooltip=[
                    alt.Tooltip("tier:N", title="Tier"),
                    alt.Tooltip("token_fertility:Q", title="Token fertility", format=".3f"),
                    alt.Tooltip("mean_cosine_sim:Q", title="Mean cosine similarity", format=".4f"),
                ],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(_style_chart(scatter), width="stretch")

        st.markdown("#### 4-Tier JSR & Over-Refusal")
        grouped_rows: list[dict[str, Any]] = []
        chart_series = (
            ("vanilla_jsr", "Vanilla Baseline JSR"),
            ("indicshield_jsr", "IndicShield-RepE JSR"),
            ("over_refusal_rate", "Over-Refusal Rate"),
        )
        for row in metrics_frame.to_dict(orient="records"):
            for column, label in chart_series:
                grouped_rows.append(
                    {
                        "tier": row["tier"],
                        "series": label,
                        "rate_pct": _percentage_number(row[column]),
                    }
                )
        grouped_bars = (
            alt.Chart(pd.DataFrame(grouped_rows))
            .mark_bar()
            .encode(
                x=alt.X("tier:N", title="Language tier", sort=list(TIERS)),
                xOffset=alt.XOffset("series:N", sort=[label for _, label in chart_series]),
                y=alt.Y("rate_pct:Q", title="Rate (%)"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=[label for _, label in chart_series],
                        range=["#38BDF8", "#10B981", "#F43F5E"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("tier:N", title="Tier"),
                    alt.Tooltip("series:N", title="Metric"),
                    alt.Tooltip("rate_pct:Q", title="Rate (%)", format=".1f"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(_style_chart(grouped_bars), width="stretch")
    else:
        st.info("No measured metrics.csv rows are available for tier comparison.")

    drift_column, alpha_column = st.columns(2)
    with drift_column:
        st.markdown("#### Multi-Turn Drift")
        if drift_frame is None or drift_frame.empty:
            st.info("No measured multiturn_drift.csv rows are available.")
        else:
            drift = drift_frame.copy()
            drift["turn"] = pd.to_numeric(drift["turn"], errors="coerce")
            drift["jsr_pct"] = drift["jsr_pct"].map(_percentage_number)
            drift["mean_cosine_sim"] = pd.to_numeric(
                drift["mean_cosine_sim"], errors="coerce"
            )
            drift = drift.dropna(subset=["turn", "jsr_pct", "mean_cosine_sim"])
            turn_axis = alt.X("turn:Q", title="Turn", axis=alt.Axis(format="d"))
            jsr_line = (
                alt.Chart(drift)
                .mark_line(point=True, color="#38BDF8")
                .encode(
                    x=turn_axis,
                    y=alt.Y("jsr_pct:Q", title="JSR (%)"),
                    tooltip=[
                        alt.Tooltip("turn:Q", title="Turn", format="d"),
                        alt.Tooltip("jsr_pct:Q", title="JSR (%)", format=".1f"),
                    ],
                )
            )
            cosine_line = (
                alt.Chart(drift)
                .mark_line(point=True, color="#F59E0B")
                .encode(
                    x=turn_axis,
                    y=alt.Y("mean_cosine_sim:Q", title="Mean cosine similarity"),
                    tooltip=[
                        alt.Tooltip("turn:Q", title="Turn", format="d"),
                        alt.Tooltip("mean_cosine_sim:Q", title="Mean cosine", format=".4f"),
                    ],
                )
            )
            drift_chart = (
                alt.layer(jsr_line, cosine_line)
                .resolve_scale(y="independent")
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(_style_chart(drift_chart), width="stretch")

    with alpha_column:
        st.markdown("#### Alpha Sweep")
        if alpha_frame is None or alpha_frame.empty:
            st.info("No measured alpha_sweep.csv rows are available.")
        else:
            alpha_data = alpha_frame.copy()
            alpha_data["alpha"] = pd.to_numeric(alpha_data["alpha"], errors="coerce")
            alpha_data["jsr_pct"] = alpha_data["jsr_pct"].map(_percentage_number)
            alpha_data["over_refusal_pct"] = alpha_data["over_refusal_pct"].map(
                _percentage_number
            )
            alpha_data = alpha_data.dropna(
                subset=["alpha", "jsr_pct", "over_refusal_pct"]
            )
            alpha_long = alpha_data.melt(
                id_vars=["alpha"],
                value_vars=["jsr_pct", "over_refusal_pct"],
                var_name="series",
                value_name="rate_pct",
            )
            alpha_long["series"] = alpha_long["series"].map(
                {
                    "jsr_pct": "JSR",
                    "over_refusal_pct": "Over-Refusal",
                }
            )
            alpha_chart = (
                alt.Chart(alpha_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("alpha:Q", title="Steering strength (α)"),
                    y=alt.Y("rate_pct:Q", title="Rate (%)"),
                    color=alt.Color(
                        "series:N",
                        title=None,
                        scale=alt.Scale(
                            domain=["JSR", "Over-Refusal"],
                            range=["#10B981", "#F43F5E"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("alpha:Q", title="α", format=".1f"),
                        alt.Tooltip("series:N", title="Metric"),
                        alt.Tooltip("rate_pct:Q", title="Rate (%)", format=".1f"),
                    ],
                )
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(_style_chart(alpha_chart), width="stretch")


st.title("IndicShield-RepE · Research Diagnostic Inspector")
st.caption("The Big Hack 2.0 · Track 03: AI Research · Qwen/Qwen2.5-3B-Instruct")

live_ready, unavailable_reasons = _live_prerequisites()
if "use_mock" not in st.session_state:
    st.session_state["use_mock"] = not live_ready
elif not live_ready and not st.session_state["use_mock"]:
    st.session_state["use_mock"] = True
USE_MOCK = st.sidebar.toggle(
    "USE_MOCK · safe preview mode",
    key="use_mock",
    disabled=not live_ready,
    help="Live inference requires a local refusal vector, the hook, cached model weights, and CUDA.",
)
if USE_MOCK:
    st.sidebar.info("Mock/preview mode: no model output or synthetic benchmark metric is generated.")
if unavailable_reasons:
    st.sidebar.warning("Live mode unavailable: " + "; ".join(unavailable_reasons) + ".")
elif not USE_MOCK:
    st.sidebar.success("Live hook prerequisites found; inference runs only when requested.")

st.sidebar.markdown("#### Result CSVs")
for filename in ("eval_logs.csv", "metrics.csv", "alpha_sweep.csv", "multiturn_drift.csv"):
    path = RESULTS_DIR / filename
    state = "rows present" if path.is_file() and path.stat().st_size > 0 else "missing or empty"
    st.sidebar.caption(f"{filename}: {state}")
st.sidebar.caption("CSV provenance is not independently verified by the dashboard.")

eval_frame, eval_error = _read_result_csv("eval_logs.csv", EVAL_LOG_COLUMNS)
eval_rows = [] if eval_frame is None else eval_frame.fillna("").to_dict(orient="records")
eval_rows_by_id = {str(row["id"]): row for row in eval_rows}
if eval_error:
    st.error(eval_error)
for state_key, default in (
    ("research_prompt", ""),
    ("research_tier", TIERS[0]),
    ("reviewed_benign", False),
    ("measured_case_id", ""),
):
    if state_key not in st.session_state:
        st.session_state[state_key] = default

if eval_rows:
    st.selectbox(
        "Measured prompt from eval_logs.csv",
        options=["", *eval_rows_by_id],
        format_func=lambda case_id: (
            "Manual prompt (no measured row)"
            if not case_id
            else (
                f"{eval_rows_by_id[case_id]['id']} · {eval_rows_by_id[case_id]['tier']} · "
                f"{eval_rows_by_id[case_id]['prompt']}"
            )
        ),
        key="measured_case_id",
        on_change=_sync_measured_selection,
        args=(eval_rows_by_id,),
    )
else:
    st.caption("No measured prompt rows are available in eval_logs.csv.")

prompt = st.text_area(
    "Research prompt",
    placeholder="Enter a safe research-evaluation prompt.",
    height=110,
    key="research_prompt",
)
tier = st.selectbox(
    "Language tier",
    TIERS,
    key="research_tier",
)
reviewed_benign = st.checkbox(
    "This is a reviewed benign control prompt",
    help="Unreviewed prompts are treated as harmful for display masking.",
    key="reviewed_benign",
)
is_harmful = not reviewed_benign

tau_column, alpha_column = st.columns(2)
with tau_column:
    tau = st.slider(
        "Gate threshold (tau)",
        min_value=0.05,
        max_value=0.40,
        value=0.18,
        step=0.005,
        key="gate_threshold",
    )
with alpha_column:
    alpha = st.slider("Steering strength (alpha)", min_value=0.0, max_value=3.0, value=1.6, step=0.1)

measured_prompt_row = next(
    (
        row
        for row in eval_rows
        if str(row["tier"]) == tier and str(row["prompt"]).strip() == prompt.strip()
    ),
    None,
)
if measured_prompt_row is not None:
    selected_similarity = float(measured_prompt_row["cosine_sim"])
    if selected_similarity >= tau:
        st.success("TRIGGERED — Steering Gate Opens")
    else:
        st.warning("BELOW THRESHOLD — Pass-Through")
    st.caption(
        f"Measured cosine_sim {selected_similarity:.4f} compared with selected τ {tau:.3f} "
        "(triggered when cosine_sim ≥ τ)."
    )

if st.button("Run research diagnostic", type="primary", disabled=not bool(prompt.strip())):
    if measured_prompt_row is not None:
        st.session_state["diagnostic_result"] = _measured_result(measured_prompt_row, tau)
    elif USE_MOCK:
        st.session_state["diagnostic_result"] = {
            "mode": "preview",
            "prompt": prompt,
            "is_harmful": is_harmful,
            "tau": tau,
            "alpha": alpha,
            "message": "Preview mode only; inference was not run and no response was generated.",
        }
    else:
        try:
            live_result = _run_live(prompt, tau, alpha, is_harmful)
            live_result["mode"] = "live"
            live_result["prompt"] = prompt
            live_result["is_harmful"] = is_harmful
            live_result["tau"] = tau
            live_result["alpha"] = alpha
            st.session_state["diagnostic_result"] = live_result
        except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            st.session_state["diagnostic_result"] = {
                "mode": "fallback",
                "prompt": prompt,
                "is_harmful": is_harmful,
                "tau": tau,
                "alpha": alpha,
                "message": f"Live inference failed; no response was shown. Preview fallback: {error}",
            }

result = st.session_state.get("diagnostic_result")
if result is not None and (
    result.get("prompt") != prompt
    or result.get("is_harmful") != is_harmful
    or result.get("tau") != tau
    or (result.get("mode") in ("preview", "fallback", "live") and result.get("alpha") != alpha)
    or (result.get("mode") == "live" and USE_MOCK)
    or (result.get("mode") in ("preview", "fallback") and not USE_MOCK)
):
    result = None
if measured_prompt_row is not None:
    result = _measured_result(measured_prompt_row, tau)
vanilla_column, shield_column = st.columns(2)
with vanilla_column:
    st.subheader("Vanilla baseline")
    if result is None:
        st.info("Run a diagnostic to display a baseline response.")
    elif result["mode"] in ("live", "measured"):
        st.write(result["vanilla_display"])
        if result["mode"] == "measured":
            st.caption(f"Measured CSV row · {result['id']} · {result['tier']}")
        else:
            st.caption("Live model generation · output masked when unrefused and prompt is unreviewed.")
        if result["vanilla_refused"]:
            st.success("Vanilla refused")
        else:
            st.warning("Vanilla did not refuse")
    else:
        st.info(result["message"])
with shield_column:
    st.subheader("IndicShield-RepE")
    if result is None:
        st.info("Run a diagnostic to display the hooked response.")
    elif result["mode"] in ("live", "measured"):
        st.write(result["indicshield_display"])
        if result["mode"] == "measured":
            st.caption(f"Measured CSV row · {result['id']} · {result['tier']}")
        else:
            st.caption("Live hook output · output masked when unrefused and prompt is unreviewed.")
        if result["indicshield_refused"]:
            st.success("IndicShield refused")
        else:
            st.warning("IndicShield did not refuse")
        similarity, latency = st.columns(2)
        similarity.metric(
            "Prefill cosine similarity",
            "—" if result["cosine_similarity"] is None else f"{result['cosine_similarity']:.4f}",
        )
        latency.metric(
            "Hook latency",
            "—" if result["latency_ms"] is None else f"{result['latency_ms']:.2f} ms",
        )
        if result["mode"] == "measured":
            if result["triggered"]:
                st.success("TRIGGERED — Steering Gate Opens")
            else:
                st.warning("BELOW THRESHOLD — Pass-Through")
            st.caption("Gate status is derived from the logged cosine_sim and selected τ.")
        else:
            st.metric("Hook triggered", "Yes" if result["triggered"] else "No")
    else:
        st.info(result["message"])

st.divider()
st.subheader("Benchmark CSV snapshot")
metrics_rows, metrics_error = _read_metrics()
if metrics_error:
    st.error(metrics_error)
elif metrics_rows is not None and not metrics_rows.empty:
    st.caption("Local CSV values · provenance not independently verified · not a live inference result.")
    st.dataframe(
        _format_rate_columns(
            metrics_rows,
            ("vanilla_jsr", "indicshield_jsr", "over_refusal_rate"),
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.info("No non-empty metrics.csv is available; no benchmark values are fabricated.")

st.subheader("Failure analysis CSVs")
alpha_frame, alpha_error = _read_result_csv("alpha_sweep.csv", ALPHA_COLUMNS)
drift_frame, drift_error = _read_result_csv("multiturn_drift.csv", MULTITURN_COLUMNS)
for filename, frame, error, rate_columns in (
    ("alpha_sweep.csv", alpha_frame, alpha_error, ("jsr_pct", "over_refusal_pct")),
    ("multiturn_drift.csv", drift_frame, drift_error, ("jsr_pct",)),
):
    st.markdown(f"**{filename}**")
    if error:
        st.error(error)
    elif frame is None or frame.empty:
        st.info(f"No non-empty {filename} is available.")
    else:
        st.dataframe(
            _format_rate_columns(frame, rate_columns),
            hide_index=True,
            width="stretch",
        )

_render_interactive_charts(
    eval_frame,
    metrics_rows,
    alpha_frame,
    drift_frame,
    tau,
)
