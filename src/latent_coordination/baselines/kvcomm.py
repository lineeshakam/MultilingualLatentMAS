"""KVComm Baseline: online cross-context KV-cache communication.

Simplified implementation of the KVComm approach (arXiv:2510.12872, "online
cross-context KV-cache communication"). dev_doc.md §3 names this baseline
from a one-line description only — no paper text was available when this was
written, so fidelity to the original method is best-effort, not verified.
Treat this as a reasonable-faithful reference implementation, not a
reproduction.

Key idea distinguishing KVComm from :class:`CacheToCacheBaseline` (the
existing, *static* pairwise KV-fusion baseline in ``cache_to_cache.py``):
KVComm is "online" — communication happens live, during decode, by splicing
a fused KV pair into the receiver's cache at prefill time via a forward hook,
rather than being computed once and passed as a plain tensor after the fact.
This module reuses C2C's per-pair learned-projection idiom for the actual
fusion math (same shape conventions), and adds the hook plumbing modeled on
``base_agent.py``'s ``_make_injection_hook``/``generate_and_capture`` pattern
that this repo already uses for live latent injection during generation.

Firewall note (strategy.md §6 / dev_doc.md §1): no SVD/CLAP machinery here —
only linear projections and elementwise gating.

Reference:
    KVComm: Online Cross-Context KV-Cache Communication for Efficient
    Multi-Agent Reasoning. arXiv:2510.12872.
"""


import logging
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


class _KVCommProjection(nn.Module):
    """Per-pair learned projection + gate, shared across all layers.

    A genuinely per-layer projection (a distinct module per transformer
    layer) would scale parameters by num_layers per registered pair; this
    baseline keeps one projection per (sender, receiver) pair and reuses it
    at every layer, trading some fidelity for the same O(N²)-in-agents (not
    O(N²·layers)) footprint as :class:`CacheToCacheBaseline`.
    """

    def __init__(self, sender_dim: int, receiver_dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(sender_dim, receiver_dim, bias=False)
        self.val_proj = nn.Linear(sender_dim, receiver_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, sender_k: Tensor, sender_v: Tensor, receiver_k: Tensor, receiver_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fuse sender KV into receiver KV, shapes (B, L, D)."""
        g = torch.sigmoid(self.gate)
        s_k = self.key_proj(sender_k.float().mean(dim=1, keepdim=True))
        s_v = self.val_proj(sender_v.float().mean(dim=1, keepdim=True))
        fused_k = receiver_k + g * s_k.expand_as(receiver_k)
        fused_v = receiver_v + g * s_v.expand_as(receiver_v)
        return fused_k, fused_v


class KVCommBaseline:
    """Online cross-context KV-cache communication.

    Args:
        device: PyTorch device string.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._agent_dims: Dict[str, int] = {}
        self._projections: Dict[Tuple[str, str], _KVCommProjection] = {}
        self._fuse_count = 0
        logger.info("KVCommBaseline initialized (online, per-pair projection reused across layers)")

    def register_agent(self, agent_id: str, num_heads: int, head_dim: int) -> None:
        """Register an agent's KV-cache dimension (num_heads * head_dim).

        Args:
            agent_id: Unique agent identifier.
            num_heads: Number of attention heads.
            head_dim: Per-head dimension.
        """
        kv_dim = num_heads * head_dim
        self._agent_dims[agent_id] = kv_dim
        for other_id, other_dim in self._agent_dims.items():
            if other_id == agent_id:
                continue
            fwd = (agent_id, other_id)
            if fwd not in self._projections:
                self._projections[fwd] = _KVCommProjection(kv_dim, other_dim).to(self.device)
            bwd = (other_id, agent_id)
            if bwd not in self._projections:
                self._projections[bwd] = _KVCommProjection(other_dim, kv_dim).to(self.device)
        logger.info("KVCommBaseline: registered '%s' (num_heads=%d, head_dim=%d)", agent_id, num_heads, head_dim)

    def fuse(
        self,
        sender_id: str,
        receiver_id: str,
        sender_k: Tensor,
        sender_v: Tensor,
        receiver_k: Tensor,
        receiver_v: Tensor,
        layer_idx: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Fuse sender KV into receiver KV cache at a given layer.

        ``layer_idx`` is bookkeeping only (see :func:`communication_stats`);
        the projection module itself is shared across layers.

        Args:
            sender_id: Agent providing context.
            receiver_id: Agent receiving and integrating context.
            sender_k: Sender key cache, shape (B, L_s, D_s).
            sender_v: Sender value cache, shape (B, L_s, D_s).
            receiver_k: Receiver key cache, shape (B, L_r, D_r).
            receiver_v: Receiver value cache, shape (B, L_r, D_r).
            layer_idx: Transformer layer index this fusion applies to.

        Returns:
            Tuple of fused (key, value) caches, shape (B, L_r, D_r).
        """
        proj = self._get_projection(sender_id, receiver_id)
        self._fuse_count += 1
        return proj(
            sender_k.to(self.device), sender_v.to(self.device),
            receiver_k.to(self.device), receiver_v.to(self.device),
        )

    def build_kv_injection_hook(
        self,
        sender_id: str,
        receiver_id: str,
        sender_kv_provider: Callable[[], Tuple[Tensor, Tensor]],
        layer_idx: int = 0,
    ) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Build a live prefill-time KV-splicing hook.

        Modeled on ``base_agent.py``'s injection-hook pattern (soft-prefix
        injection at prefill only, not re-applied every decode step): the
        returned callable is meant to be invoked once, at the receiver's
        prefill step, with the receiver's own freshly-computed (key, value)
        for ``layer_idx`` — it calls ``sender_kv_provider()`` to fetch the
        sender's current KV pair and returns the fused pair to splice into
        the receiver's cache in place of the raw one.

        Args:
            sender_id: Agent whose context is being shared.
            receiver_id: Agent whose cache is being spliced.
            sender_kv_provider: Zero-arg callable returning the sender's
                current ``(key, value)`` tensors, shape (B, L_s, D_s) each.
            layer_idx: Layer this hook applies to.

        Returns:
            ``hook(receiver_k, receiver_v) -> (fused_k, fused_v)``.
        """
        self._require_agent(sender_id)
        self._require_agent(receiver_id)

        def hook(receiver_k: Tensor, receiver_v: Tensor) -> Tuple[Tensor, Tensor]:
            sender_k, sender_v = sender_kv_provider()
            return self.fuse(sender_id, receiver_id, sender_k, sender_v, receiver_k, receiver_v, layer_idx)

        return hook

    def communication_stats(self) -> Dict[str, int]:
        """Report cumulative fuse-call statistics."""
        return {
            "n_registered_agents": len(self._agent_dims),
            "n_projection_pairs": len(self._projections),
            "n_fuse_calls": self._fuse_count,
        }

    def _get_projection(self, sender_id: str, receiver_id: str) -> _KVCommProjection:
        key = (sender_id, receiver_id)
        if key not in self._projections:
            raise KeyError(
                f"No projection registered for ({sender_id}, {receiver_id}). "
                f"Call register_agent() for both before calling fuse()."
            )
        return self._projections[key]

    def _require_agent(self, agent_id: str) -> None:
        if agent_id not in self._agent_dims:
            raise KeyError(f"Agent '{agent_id}' not registered. Call register_agent() first.")
