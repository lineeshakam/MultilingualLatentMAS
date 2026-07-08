"""DyTopo Baseline: dynamic semantic-similarity topology routing.

Simplified implementation of the DyTopo approach (arXiv:2602.06039, "dynamic
semantic-similarity topology routing"). dev_doc.md §3 names this baseline
from a one-line description only — no paper text was available when this was
written, so fidelity to the original method is best-effort, not verified.

Key idea distinguishing DyTopo from :class:`GDesignerBaseline`
(``gdesigner_mas_router.py``): G-Designer decodes a query-conditioned
adjacency once, from a *trained* VGAE, and holds it fixed. DyTopo's "dynamic"
claim requires *recomputing* the adjacency every communication round from the
agents' *live* hidden states — no training, no learned parameters, just
pairwise cosine similarity between current agent states, thresholded into a
binary graph, with an optional per-round threshold anneal. This baseline
reuses G-Designer's binary-adjacency tensor shape convention but replaces its
VGAE inner-product decoder with plain, untrained cosine similarity.

Firewall note (strategy.md §6 / dev_doc.md §1): cosine similarity only — no
SVD/CLAP decomposition (that machinery is reserved for
``src/mechanistic_disentangle/``).

Reference:
    DyTopo: Dynamic Semantic-Similarity Topology Routing for Multi-Agent
    Systems. arXiv:2602.06039.
"""


import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
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


class DyTopoBaseline:
    """Recomputes a communication topology each round from live agent-state similarity.

    Args:
        device: PyTorch device string.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._history: List[Tensor] = []
        logger.info("DyTopoBaseline initialized (untrained, recomputed per round)")

    def compute_topology(
        self,
        agent_states: Dict[str, Tensor],
        round_idx: int = 0,
        threshold: float = 0.5,
        decay: float = 0.0,
    ) -> Tensor:
        """Compute a binary adjacency matrix from live pairwise cosine similarity.

        Args:
            agent_states: Mapping agent_id -> hidden state, each shape (D,)
                or (B, D). All agents must share the same D.
            round_idx: Current communication round (0-indexed); used with
                ``decay`` to anneal the threshold over rounds.
            threshold: Base similarity threshold above which an edge exists.
            decay: Per-round threshold decrease (``effective = threshold -
                decay * round_idx``, clamped to [0, 1]); 0.0 disables anneal.

        Returns:
            Binary adjacency, shape (N, N), agent order = ``list(agent_states)``.
            ``adj[i, j] == 1`` means agent i's state is similar enough to
            agent j's to route a message i -> j (diagonal is always 0 —
            no self-loops).
        """
        if not agent_states:
            raise ValueError("compute_topology requires at least one agent state.")
        ids = list(agent_states.keys())
        states = []
        for aid in ids:
            s = agent_states[aid]
            if s.dim() == 2:
                s = s.mean(dim=0)
            states.append(s.to(self.device).float())
        stacked = torch.stack(states, dim=0)  # (N, D)

        norm = F.normalize(stacked, dim=-1)
        sim = norm @ norm.T  # (N, N) cosine similarity

        eff_threshold = max(0.0, min(1.0, threshold - decay * round_idx))
        adj = (sim > eff_threshold).float()
        adj.fill_diagonal_(0.0)

        self._history.append(adj)
        return adj

    def topology_diversity_stats(self) -> Dict[str, float]:
        """Measure how much the adjacency actually changed round-to-round.

        Returns:
            Dict with ``n_rounds``, ``mean_edge_count``, and
            ``mean_hamming_delta`` (average number of differing edges between
            consecutive rounds — 0.0 if fewer than 2 rounds recorded, i.e.
            the topology has never been shown to be "dynamic").
        """
        if not self._history:
            return {"n_rounds": 0, "mean_edge_count": 0.0, "mean_hamming_delta": 0.0}
        edge_counts = [float(a.sum().item()) for a in self._history]
        deltas = []
        for prev, cur in zip(self._history[:-1], self._history[1:]):
            deltas.append(float((prev != cur).float().sum().item()))
        return {
            "n_rounds": len(self._history),
            "mean_edge_count": sum(edge_counts) / len(edge_counts),
            "mean_hamming_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        }

    def reset_history(self) -> None:
        """Clear recorded round history (e.g. between tasks)."""
        self._history = []
