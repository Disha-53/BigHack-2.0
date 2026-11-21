"""Extract and save a refusal direction from model hidden-state activations."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
FIRST_LAYER = 12
LAST_LAYER = 24
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "english_pairs.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "refusal_vec.pt"


def load_model(
    model_name: str = MODEL_NAME,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the research model and tokenizer using float16 weights."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def extract_last_token_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    layers: Sequence[int] = tuple(range(FIRST_LAYER, LAST_LAYER + 1)),
) -> torch.Tensor:
    """Return last-token states shaped [prompts, layers, hidden]."""
    if not prompts:
        raise ValueError("At least one prompt is required to extract activations.")
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("At least one non-negative layer index is required.")

    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    encoded = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_device = model.get_input_embeddings().weight.device
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)

    if outputs.hidden_states is None or len(outputs.hidden_states) <= max(layers) + 1:
        raise ValueError(f"The model must expose hidden states through layer {max(layers)}.")

    positions = torch.arange(encoded["attention_mask"].shape[1], device=input_device)
    last_positions = (encoded["attention_mask"] * positions).max(dim=1).values
    batch_indices = torch.arange(len(prompts), device=input_device)
    states = [
        outputs.hidden_states[layer + 1][batch_indices, last_positions]
        for layer in layers
    ]
    return torch.stack(states, dim=1).float().cpu()


def compute_refusal_vector(
    refusal_activations: torch.Tensor,
    non_refusal_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute a normalized difference-in-means vector across layers 12–24.

    Inputs may be [examples, hidden] or [examples, layers, hidden]. For 3-D
    inputs the mean difference is averaged across layers to produce one vector.
    """
    if refusal_activations.ndim not in (2, 3):
        raise ValueError("Refusal activations must have 2 or 3 dimensions.")
    if non_refusal_activations.ndim != refusal_activations.ndim:
        raise ValueError("Activation tensors must have the same number of dimensions.")
    if refusal_activations.shape[1:] != non_refusal_activations.shape[1:]:
        raise ValueError("Activation tensors must have matching layer and hidden dimensions.")
    if refusal_activations.shape[0] == 0 or non_refusal_activations.shape[0] == 0:
        raise ValueError("Both activation groups must contain at least one example.")

    difference = refusal_activations.float().mean(dim=0) - non_refusal_activations.float().mean(dim=0)
    if difference.ndim == 2:
        difference = difference.mean(dim=0)
    norm = torch.linalg.vector_norm(difference)
    if norm.item() == 0:
        raise ValueError("The mean activation difference is zero; it cannot be normalized.")
    return difference / norm


def save_refusal_vector(refusal_vec: torch.Tensor, output_path: Path = OUTPUT_PATH) -> None:
    """Save a normalized refusal vector, creating its parent directory as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refusal_vec.detach().cpu(), output_path)


def extract_refusal_vectors(
    model_name: str = MODEL_NAME,
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
    layers: Sequence[int] = tuple(range(FIRST_LAYER, LAST_LAYER + 1)),
) -> dict[str, Any]:
    """Extract normalized layer-wise refusal vectors and select the best AUROC layer."""
    with data_path.open(encoding="utf-8") as dataset_file:
        pairs = json.load(dataset_file)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Expected a non-empty JSON array of prompt pairs in {data_path}.")
    for index, pair in enumerate(pairs):
        if (
            not isinstance(pair, dict)
            or not isinstance(pair.get("harmful"), str)
            or not pair["harmful"].strip()
            or not isinstance(pair.get("harmless"), str)
            or not pair["harmless"].strip()
        ):
            raise ValueError(
                f"Prompt pair {index} in {data_path} must contain non-empty "
                "'harmful' and 'harmless' strings."
            )
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 for layer in layers):
        raise ValueError("Layer indices must be distinct, non-negative integers.")

    print(f"Loading model: {model_name} (torch.float16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    harmful_acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    harmless_acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for pair in tqdm(pairs, desc="Extracting hidden states"):
        states = extract_last_token_hidden_states(
            model,
            tokenizer,
            [pair["harmful"], pair["harmless"]],
            layers,
        )
        for layer_position, layer in enumerate(layers):
            harmful_acts[layer].append(states[0, layer_position])
            harmless_acts[layer].append(states[1, layer_position])

    vectors: dict[int, torch.Tensor] = {}
    auroc_scores: dict[int, float] = {}
    header = f"{'Layer':>5} | {'AUROC':>7} | {'Mean Harmful Sim':>16} | {'Mean Harmless Sim':>17}"
    print("\n" + header)
    print("-" * len(header))

    for layer in layers:
        harmful_tensor = torch.stack(harmful_acts[layer])
        harmless_tensor = torch.stack(harmless_acts[layer])
        difference = harmful_tensor.mean(dim=0) - harmless_tensor.mean(dim=0)
        norm = torch.linalg.vector_norm(difference)
        if norm.item() == 0:
            raise ValueError(f"The mean activation difference is zero at layer {layer}.")
        vector = F.normalize(difference, dim=-1)
        vectors[layer] = vector

        harmful_similarities = F.cosine_similarity(harmful_tensor, vector.unsqueeze(0), dim=-1)
        harmless_similarities = F.cosine_similarity(harmless_tensor, vector.unsqueeze(0), dim=-1)
        labels = [1] * len(harmful_similarities) + [0] * len(harmless_similarities)
        scores = torch.cat([harmful_similarities, harmless_similarities]).tolist()
        auroc = float(roc_auc_score(labels, scores))
        auroc_scores[layer] = auroc
        print(
            f"{layer:>5} | {auroc:>7.4f} | "
            f"{harmful_similarities.mean().item():>16.4f} | "
            f"{harmless_similarities.mean().item():>17.4f}"
        )

    best_layer = max(auroc_scores, key=auroc_scores.__getitem__)
    print(f"\nSelected best_layer = {best_layer} (AUROC = {auroc_scores[best_layer]:.4f})")
    payload: dict[str, Any] = {
        "vectors": vectors,
        "best_layer": best_layer,
        "auroc": auroc_scores,
        "model_name": model_name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved refusal vectors to {output_path}")
    return payload


def main() -> None:
    """Run layer-wise refusal-vector extraction from paired English prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        "--model-name",
        default=MODEL_NAME,
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--data_path",
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="JSON file containing harmful/harmless prompt pairs.",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for the saved refusal-vector dictionary.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(FIRST_LAYER, LAST_LAYER + 1)),
        help="Layer indices to probe (default: 12..24).",
    )
    args = parser.parse_args()
    extract_refusal_vectors(
        model_name=args.model_name,
        data_path=args.data_path,
        output_path=args.output_path,
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
