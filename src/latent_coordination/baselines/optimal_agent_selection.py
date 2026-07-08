"""Optimal-Agent-Selection Baseline: cost-budgeted utility-maximizing agent subset.

Simplified implementation of the Optimal-Agent-Selection approach
(arXiv:2511.02200). dev_doc.md §3 names this baseline from a one-line
description only — no paper text was available when this was written, so
fidelity to the original method is best-effort, not verified. Unlike KVComm
and DyTopo, no existing baseline in this repo is structurally close: the
nearest idiom is :class:`MasRouterBaseline`'s threshold-selection return-dict
(``gdesigner_mas_router.py``), but that selects fixed *roles* from a learned
cascade, not a cost-optimized *subset* of candidate agents.

Given this pipeline's real agent pool is small (max ~3-4 roles, per
``dev_doc.md``'s "Hardware Constraints" note — one agent per device, no
`parallel_agents`), an *exact* 0/1-knapsack-style subset search over all
``2**N`` candidate subsets is honest and tractable: no fabricated ML, no
approximation heuristics standing in for real optimization at a scale this
system never actually reaches.

Firewall note (strategy.md §6 / dev_doc.md §1): no SVD/CLAP machinery here —
plain combinatorial search over real (utility, cost) inputs.

Reference:
    Optimal Agent Selection for Cost-Constrained Multi-Agent Systems.
    arXiv:2511.02200.
"""


import itertools
import logging
from typing import Dict, List

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)

_MAX_EXACT_CANDIDATES = 20  # 2**20 subsets is still fast; guards against misuse at scale.


class OptimalAgentSelectionBaseline:
    """Selects the utility-maximizing agent subset under a hard cost budget."""

    def __init__(self) -> None:
        self._last_rationale: Dict[str, object] = {}
        logger.info("OptimalAgentSelectionBaseline initialized (exact subset search)")

    def select_agents(
        self, candidates: Dict[str, Dict[str, float]], budget: float,
    ) -> Dict[str, object]:
        """Select the subset of candidate agents maximizing total utility under budget.

        Args:
            candidates: Mapping agent_id -> {"utility": float, "cost": float}.
                Both must be present and cost must be non-negative.
            budget: Maximum total cost of the selected subset (inclusive).

        Returns:
            Dict with ``selected_agents`` (List[str]), ``total_utility``
            (float), ``total_cost`` (float), and ``n_candidates_considered``.

        Raises:
            ValueError: If a candidate is missing ``utility``/``cost``, cost
                is negative, or there are more than
                :data:`_MAX_EXACT_CANDIDATES` candidates (exact search would
                be impractical — this pipeline never has that many real
                agents, so hitting this is a misuse signal, not a case to
                silently approximate).
        """
        if len(candidates) > _MAX_EXACT_CANDIDATES:
            raise ValueError(
                f"select_agents got {len(candidates)} candidates; exact search caps at "
                f"{_MAX_EXACT_CANDIDATES}. This pipeline's real agent pool is far smaller "
                f"(max ~3-4 roles) -- this is a misuse signal, not a scale to approximate."
            )
        ids: List[str] = list(candidates.keys())
        for aid in ids:
            c = candidates[aid]
            if "utility" not in c or "cost" not in c:
                raise ValueError(f"Candidate '{aid}' missing 'utility' or 'cost' field.")
            if c["cost"] < 0:
                raise ValueError(f"Candidate '{aid}' has negative cost ({c['cost']}); not a valid budget input.")

        best_subset: tuple = ()
        best_utility = -1.0
        best_cost = 0.0
        for r in range(len(ids) + 1):
            for subset in itertools.combinations(ids, r):
                total_cost = sum(candidates[a]["cost"] for a in subset)
                if total_cost > budget:
                    continue
                total_utility = sum(candidates[a]["utility"] for a in subset)
                if total_utility > best_utility:
                    best_utility = total_utility
                    best_subset = subset
                    best_cost = total_cost

        if best_utility < 0:
            # Only reachable if even the empty subset's cost (0.0) exceeds a
            # negative budget -- an invalid budget, not a "no agents fit" case.
            raise ValueError(f"No feasible subset (including empty) fits budget={budget}.")

        result = {
            "selected_agents": list(best_subset),
            "total_utility": float(best_utility),
            "total_cost": float(best_cost),
            "n_candidates_considered": len(ids),
        }
        self._last_rationale = {
            **result,
            "budget": budget,
            "unselected_agents": [a for a in ids if a not in best_subset],
        }
        return result

    def selection_rationale(self) -> Dict[str, object]:
        """Return the rationale dict from the most recent :func:`select_agents` call."""
        if not self._last_rationale:
            raise RuntimeError("No selection has been made yet; call select_agents() first.")
        return self._last_rationale
