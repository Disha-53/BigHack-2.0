"""Extract and save a refusal direction from model hidden-state activations."""

from pathlib import Path
from typing import Sequence

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


def save_refusal_vector(refusal_vec: torch.Tensor, output_path: Path = OUTPUT_PATH) -> None:
    """Save a normalized refusal vector, creating its parent directory as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(refusal_vec.detach().cpu(), output_path)


def main() -> None:
    """Report the remaining dataset-specific extraction step."""
    raise NotImplementedError(
        "Load refusal and non-refusal prompt groups, extract their activations, "
        "then call compute_refusal_vector() and save_refusal_vector()."
    )


if __name__ == "__main__":
    main()
