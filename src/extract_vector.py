"""Extract and save a refusal direction from model hidden-state activations."""

from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
FIRST_LAYER = 12
LAST_LAYER = 24
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "refusal_vec.pt"


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
) -> torch.Tensor:
    """Return last-token states for layers 12–24, shaped [prompts, layers, hidden]."""
    if not prompts:
        raise ValueError("At least one prompt is required to extract activations.")

    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_device = model.get_input_embeddings().weight.device
    encoded = {name: tensor.to(input_device) for name, tensor in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)

    if outputs.hidden_states is None or len(outputs.hidden_states) <= LAST_LAYER:
        raise ValueError(f"The model must expose hidden states through layer {LAST_LAYER}.")

    last_positions = encoded["attention_mask"].sum(dim=1) - 1
    batch_indices = torch.arange(len(prompts), device=input_device)
    states = [
        outputs.hidden_states[layer][batch_indices, last_positions]
        for layer in range(FIRST_LAYER, LAST_LAYER + 1)
    ]
    return torch.stack(states, dim=1)


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


def compute_refusal_vectors(
    refusal_activations: torch.Tensor,
    non_refusal_activations: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Compute one normalized refusal direction for each layer from 12 to 24."""
    if refusal_activations.ndim != 3 or non_refusal_activations.ndim != 3:
        raise ValueError("Per-layer refusal activations must have three dimensions.")
    if refusal_activations.shape[1:] != non_refusal_activations.shape[1:]:
        raise ValueError("Activation tensors must have matching layer and hidden dimensions.")
    if refusal_activations.shape[0] == 0 or non_refusal_activations.shape[0] == 0:
        raise ValueError("Both activation groups must contain at least one example.")
    if refusal_activations.shape[1] != LAST_LAYER - FIRST_LAYER + 1:
        raise ValueError(
            f"Activations must include exactly layers {FIRST_LAYER} through {LAST_LAYER}."
        )

    differences = (
        refusal_activations.float().mean(dim=0)
        - non_refusal_activations.float().mean(dim=0)
    )
    norms = torch.linalg.vector_norm(differences, dim=-1)
    zero_layers = torch.nonzero(norms == 0, as_tuple=False).flatten().tolist()
    if zero_layers:
        layers = [FIRST_LAYER + index for index in zero_layers]
        raise ValueError(f"The mean activation difference is zero for layers: {layers}.")

    normalized = differences / norms.unsqueeze(-1)
    return {
        layer: normalized[layer - FIRST_LAYER]
        for layer in range(FIRST_LAYER, LAST_LAYER + 1)
    }


def save_refusal_vector(refusal_vec: torch.Tensor, output_path: Path = OUTPUT_PATH) -> None:
    """Save a normalized refusal vector, creating its parent directory as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refusal_vec.detach().cpu(), output_path)


def save_refusal_vectors(
    vectors: Mapping[int, torch.Tensor],
    best_layer: int,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save the per-layer artifact consumed by the live dashboard."""
    required_layers = set(range(FIRST_LAYER, LAST_LAYER + 1))
    if set(vectors) != required_layers:
        missing = sorted(required_layers - set(vectors))
        extra = sorted(set(vectors) - required_layers)
        raise ValueError(
            f"Expected vectors for layers {FIRST_LAYER}–{LAST_LAYER}; "
            f"missing={missing}, extra={extra}."
        )
    if best_layer not in required_layers:
        raise ValueError(f"best_layer must be between {FIRST_LAYER} and {LAST_LAYER}.")

    hidden_sizes = set()
    artifact_vectors: dict[int, torch.Tensor] = {}
    for layer in sorted(required_layers):
        vector = vectors[layer]
        if vector.ndim != 1:
            raise ValueError(f"Refusal vector for layer {layer} must be one-dimensional.")
        hidden_sizes.add(vector.shape[0])
        artifact_vectors[layer] = vector.detach().cpu()
    if len(hidden_sizes) != 1:
        raise ValueError("All per-layer refusal vectors must have the same hidden size.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"vectors": artifact_vectors, "best_layer": best_layer},
        output_path,
    )


def main() -> None:
    """Report the remaining dataset-specific extraction step."""
    raise NotImplementedError(
        "Load refusal and non-refusal prompt groups, extract their activations, "
        "then call compute_refusal_vectors() and save_refusal_vectors() with the "
        "selected best layer."
    )


if __name__ == "__main__":
    main()
