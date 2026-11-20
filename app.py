"""Safe Streamlit inspector for local IndicShield CSVs and optional live inference."""

import csv
import importlib.util
import subprocess
import sys
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
    return {
        "vanilla_display": redact_if_unsafe(vanilla_response, is_harmful, vanilla_refused),
        "indicshield_display": redact_if_unsafe(shield_response, is_harmful, shield_refused),
        "vanilla_refused": vanilla_refused,
        "indicshield_refused": shield_refused,
        "cosine_similarity": probe.values[0] if probe.values else None,
        "latency_ms": sum(hook.latencies_ms) if hook.latencies_ms else None,
    }


def _read_metrics() -> tuple[list[dict[str, str]], str | None]:
    """Load a usable metrics CSV, reporting missing or malformed input."""
    path = RESULTS_DIR / "metrics.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return [], None
    try:
        with path.open(encoding="utf-8-sig", newline="") as metrics_file:
            reader = csv.DictReader(metrics_file)
            if tuple(reader.fieldnames or ()) != METRICS_COLUMNS:
                return [], f"{path.name} does not match the benchmark CSV schema."
            rows = list(reader)
    except OSError as error:
        return [], f"Could not read {path.name}: {error}"
    if not rows:
        return [], None
    return rows, None


def _render_charts() -> None:
    """Display generated charts if they are available in results/."""
    chart_paths = (
        ("Token fertility and refusal similarity", RESULTS_DIR / "chart1_fertility.png"),
        ("Benchmark refusal rates", RESULTS_DIR / "chart2_benchmark.png"),
        ("Multi-turn drift and alpha sweep", RESULTS_DIR / "chart3_failure.png"),
    )
    existing = [(caption, path) for caption, path in chart_paths if path.is_file()]
    if not existing:
        st.info("Charts have not been generated. Run `python results/generate_charts.py`.")
        return
    for caption, path in existing:
        st.image(str(path), caption=caption, width="stretch")


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
for filename in ("metrics.csv", "alpha_sweep.csv", "multiturn_drift.csv"):
    path = RESULTS_DIR / filename
    state = "rows present" if path.is_file() and path.stat().st_size > 0 else "missing or empty"
    st.sidebar.caption(f"{filename}: {state}")
st.sidebar.caption("CSV provenance is not independently verified by the dashboard.")

prompt = st.text_area(
    "Research prompt",
    placeholder="Enter a safe research-evaluation prompt.",
    height=110,
)
tier = st.selectbox(
    "Language tier",
    ("english", "native_kannada", "kanglish_hinglish", "codemixed_genz_slang"),
)
reviewed_benign = st.checkbox(
    "This is a reviewed benign control prompt",
    value=False,
    help="Unreviewed prompts are treated as harmful for display masking.",
)
is_harmful = not reviewed_benign

tau_column, alpha_column = st.columns(2)
with tau_column:
    tau = st.slider("Gate threshold (tau)", min_value=-1.0, max_value=1.0, value=0.65, step=0.01)
with alpha_column:
    alpha = st.slider("Steering strength (alpha)", min_value=0.0, max_value=3.0, value=1.6, step=0.1)

if st.button("Run research diagnostic", type="primary", disabled=not bool(prompt.strip())):
    if USE_MOCK:
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
    or result.get("alpha") != alpha
    or (result.get("mode") == "live" and USE_MOCK)
):
    result = None
vanilla_column, shield_column = st.columns(2)
with vanilla_column:
    st.subheader("Vanilla baseline")
    if result is None:
        st.info("Run a diagnostic to display a baseline response.")
    elif result["mode"] == "live":
        st.write(result["vanilla_display"])
        st.caption("Live model generation · output masked when unrefused and prompt is unreviewed.")
    else:
        st.info(result["message"])
with shield_column:
    st.subheader("IndicShield-RepE")
    if result is None:
        st.info("Run a diagnostic to display the hooked response.")
    elif result["mode"] == "live":
        st.write(result["indicshield_display"])
        st.caption("Live hook output · output masked when unrefused and prompt is unreviewed.")
        similarity, latency = st.columns(2)
        similarity.metric(
            "Prefill cosine similarity",
            "—" if result["cosine_similarity"] is None else f"{result['cosine_similarity']:.4f}",
        )
        latency.metric(
            "Hook latency",
            "—" if result["latency_ms"] is None else f"{result['latency_ms']:.2f} ms",
        )
    else:
        st.info(result["message"])

st.divider()
st.subheader("Benchmark CSV snapshot")
metrics_rows, metrics_error = _read_metrics()
if metrics_error:
    st.error(metrics_error)
elif metrics_rows:
    st.caption("Local CSV values · provenance not independently verified · not a live inference result.")
    st.dataframe(metrics_rows, hide_index=True, width="stretch")
else:
    st.info("No non-empty metrics.csv is available; no benchmark values are fabricated.")

if st.button("Generate charts from result CSVs"):
    command = [
        sys.executable,
        str(RESULTS_DIR / "generate_charts.py"),
        "--results_dir",
        str(RESULTS_DIR),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        st.success(completed.stdout.strip())
        st.rerun()
    else:
        st.error(completed.stderr.strip() or completed.stdout.strip())

_render_charts()
