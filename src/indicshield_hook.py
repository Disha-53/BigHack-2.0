"""PyTorch activation hook for applying the IndicShield refusal direction."""

from time import perf_counter
from typing import Any

import torch
from torch import nn


class IndicShieldHook:
    """Apply a refusal-direction intervention during multi-token prefill calls.

    Attach the hook to the model block whose output is the residual stream at
    ``layer_idx``. Calls with a sequence length greater than one are treated as
    prefill; single-token decode calls are left untouched.
    """

    def __init__(
        self,
        refusal_vec: torch.Tensor,
        layer_idx: int,
        tau: float,
        alpha: float,
    ) -> None:
        """Initialize hook parameters and per-call latency tracking."""
        self.refusal_vec = refusal_vec.detach()
        self.layer_idx = layer_idx
        self.tau = tau
        self.alpha = alpha
        self.latencies_ms: list[float] = []
        self._handle: Any | None = None

    def attach(self, module: nn.Module) -> None:
        """Register this hook on the selected model block."""
        if self._handle is not None:
            raise RuntimeError("The IndicShield hook is already attached.")
        self._handle = module.register_forward_hook(self._forward_hook)

    def remove(self) -> None:
        """Remove the registered hook if it is attached."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        """Gate and optionally adjust a prefill residual stream."""
        del module, inputs
        started = perf_counter()
        try:
            activations, rebuild = self._get_activations(output)
            if activations.ndim != 3 or activations.shape[1] <= 1:
                return output

            vector = self.refusal_vec.to(device=activations.device, dtype=activations.dtype)
            if vector.ndim != 1 or vector.shape[0] != activations.shape[-1]:
                raise ValueError("refusal_vec must be 1-D and match the residual hidden size.")

            last_token = activations[:, -1, :]
            similarities = torch.nn.functional.cosine_similarity(
                last_token.float(),
                vector.float().expand_as(last_token),
                dim=-1,
            )
            selected = similarities > self.tau
            if not torch.any(selected):
                return output

            adjusted = activations.clone()
            adjusted[selected] = adjusted[selected] + self.alpha * vector
            return rebuild(adjusted)
        finally:
            self.latencies_ms.append((perf_counter() - started) * 1000)

    @staticmethod
    def _get_activations(output: Any) -> tuple[torch.Tensor, Any]:
        """Find the hidden-state tensor and return a function rebuilding output."""
        if isinstance(output, torch.Tensor):
            return output, lambda activations: activations
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0], lambda activations: (activations, *output[1:])
        raise TypeError("Hook output must be a tensor or a tuple beginning with a tensor.")

    @property
    def last_latency_ms(self) -> float | None:
        """Return latency for the most recent hook call, if available."""
        return self.latencies_ms[-1] if self.latencies_ms else None
