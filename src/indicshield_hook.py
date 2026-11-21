"""PyTorch activation hook for gated IndicShield refusal-direction steering."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle


class IndicShieldHook:
    """Gate residual-stream steering on cosine similarity during prompt prefill."""

    def __init__(
        self,
        refusal_vec: torch.Tensor,
        layer_idx: int,
        tau: float = 0.65,
        alpha: float = 1.6,
        steer_all_tokens: bool = True,
    ) -> None:
        if refusal_vec.ndim != 1:
            raise ValueError("refusal_vec must be a one-dimensional tensor.")
        if not torch.isfinite(refusal_vec).all():
            raise ValueError("refusal_vec must contain only finite values.")
        norm = torch.linalg.vector_norm(refusal_vec.float())
        if norm.item() == 0:
            raise ValueError("refusal_vec must have non-zero length.")
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative.")
        if not -1.0 <= tau <= 1.0:
            raise ValueError("tau must be between -1 and 1.")

        self.refusal_vec = F.normalize(refusal_vec.detach().float(), dim=-1)
        self.layer_idx = layer_idx
        self.tau = tau
        self.alpha = alpha
        self.steer_all_tokens = steer_all_tokens
        self.last_similarity = 0.0
        self.triggered = False
        self.last_latency_ms = 0.0
        self.latencies_ms: list[float] = []
        self.handle: RemovableHandle | None = None
        self._cached_vec: torch.Tensor | None = None
        self._triggered_mask: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset telemetry and per-sequence gate state before a new prompt."""
        self.last_similarity = 0.0
        self.triggered = False
        self.last_latency_ms = 0.0
        self.latencies_ms.clear()
        self._triggered_mask = None

    def _get_vec(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._cached_vec is None
            or self._cached_vec.device != device
            or self._cached_vec.dtype != dtype
        ):
            self._cached_vec = self.refusal_vec.to(device=device, dtype=dtype)
        return self._cached_vec

    def _hook_fn(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        del module, inputs
        started = time.perf_counter()
        try:
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                if not output or not isinstance(output[0], torch.Tensor):
                    raise TypeError("Hook output tuple must begin with a tensor.")
                hidden_states = output[0]
            elif isinstance(output, torch.Tensor):
                hidden_states = output
            else:
                raise TypeError("Hook output must be a tensor or a tuple beginning with a tensor.")

            if hidden_states.ndim != 3:
                raise ValueError("Hooked hidden states must have shape [batch, sequence, hidden].")
            if hidden_states.shape[-1] != self.refusal_vec.shape[0]:
                raise ValueError("refusal_vec must match the residual hidden size.")
            if hidden_states.shape[1] == 0:
                raise ValueError("Hooked hidden states must contain at least one token.")

            vector = self._get_vec(hidden_states.device, hidden_states.dtype)
            batch_size = hidden_states.shape[0]
            if hidden_states.shape[1] > 1:
                similarities = F.cosine_similarity(
                    hidden_states[:, -1, :].float(),
                    vector.float().unsqueeze(0),
                    dim=-1,
                )
                self.last_similarity = float(similarities.max().item())
                self._triggered_mask = similarities >= self.tau
                self.triggered = bool(self._triggered_mask.any().item())
            elif self._triggered_mask is None or self._triggered_mask.numel() != batch_size:
                self._triggered_mask = torch.zeros(
                    batch_size, dtype=torch.bool, device=hidden_states.device
                )
                self.triggered = False

            if self._triggered_mask is not None and self._triggered_mask.any().item():
                adjusted = hidden_states.clone()
                selected = self._triggered_mask.to(hidden_states.device)
                delta = self.alpha * vector
                if self.steer_all_tokens and hidden_states.shape[1] > 1:
                    adjusted[selected] = adjusted[selected] + delta
                else:
                    adjusted[selected, -1, :] = adjusted[selected, -1, :] + delta
                hidden_states = adjusted

            if is_tuple:
                return (hidden_states,) + output[1:]
            return hidden_states
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
            self.latencies_ms.append(self.last_latency_ms)

    @contextmanager
    def attach(self, model: nn.Module) -> Generator["IndicShieldHook", None, None]:
        """Attach to ``model.model.layers[layer_idx]`` for one generation."""
        if self.handle is not None:
            raise RuntimeError("The IndicShield hook is already attached.")
        try:
            layers = getattr(getattr(model, "model", None), "layers")
            target_layer = layers[self.layer_idx]
        except (AttributeError, IndexError, TypeError) as error:
            raise ValueError(
                "Model must expose the requested layer at model.model.layers[layer_idx]."
            ) from error

        self.reset()
        self.handle = target_layer.register_forward_hook(self._hook_fn)
        try:
            yield self
        finally:
            self.handle.remove()
            self.handle = None

    def get_stats(self) -> dict[str, Any]:
        """Return telemetry for the most recent hook invocation."""
        return {
            "similarity": round(self.last_similarity, 4),
            "triggered": self.triggered,
            "latency_ms": round(self.last_latency_ms, 4),
            "layer_idx": self.layer_idx,
            "tau": self.tau,
            "alpha": self.alpha,
        }
