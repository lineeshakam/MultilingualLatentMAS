#!/usr/bin/env python3
"""Wire ``eval/cost.py``'s CostAccountant to real, already-produced results.

dev_doc.md §11 named this "Cost Accounting" gap: ``eval/cost.py`` implements
the accuracy-vs-token-cost frontier math (``CostAccountant``, canonical
N in {4, 8, 16}) but was never called from anywhere -- fully-built dead code.

This script does NOT spin up new GPU generation. The pipeline's chain is
sequential/3-role by design (``orchestration.parallel_agents`` was removed
entirely per dev_doc.md §10 -- "there is nothing to parallelize within one
task"), so a literal N=16-agent run is not something this system can produce.
Honoring the zero-mock policy means not pretending otherwise: this script
aggregates REAL per-run cost/accuracy numbers already on disk under
``results/baselines/`` (LatentMAS/ThoughtComm, 2 real agent-calls/sample) and
``results/bench_suite/*/multiagent_benchmark_*.json`` (the coordination
pipeline's single_agent_baseline=1, token_based_mas/latent_based_mas_ours=3
real agent-calls/sample), and reports the frontier at the N values that
actually occurred (1, 2, 3) rather than fabricating 4/8/16 by extrapolation.

Usage
-----
    python scripts/run_cost_frontier.py [--output results/cost_frontier.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_coordination.eval.cost import CostAccountant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Real agent-call counts per system, per dev_doc.md §5's own cost-model table
# ("Agent calls per sample, by baseline/comm-mode"). Not configurable here --
# these are architectural facts about this codebase, not tunable parameters.
_N_AGENTS_BASELINE = 2       # LatentMASBaseline / ThoughtCommBaseline
_N_AGENTS_COMM_MODE = {
    "single_agent_baseline": 1,
    "oneflow": 1,
    "token_based_mas": 3,
    "latent_based_mas_ours": 3,
}


def _load_baseline_jsons(acct: CostAccountant) -> int:
    """results/baselines/{latentmas,thoughtcomm}/*.json -- aggregate per-run stats.

    Each file is one (benchmark, language) run's aggregate accuracy/mean-cost,
    not per-sample data, so we replay n_correct/n_total as that many
    correct/incorrect observations at the run's measured mean token cost and
    mean latency -- the most honest reconstruction possible without raw
    per-sample logs, and still strictly real numbers (no fabricated cost).
    """
    n_obs = 0
    for path in sorted(glob.glob("results/baselines/*/*.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if "n_total" not in d or "accuracy" not in d:
            continue
        system = Path(path).parent.name  # "latentmas" | "thoughtcomm"
        n_total = int(d["n_total"])
        n_correct = int(d.get("n_correct", round(d["accuracy"] * n_total)))
        mean_tokens = float(d.get("mean_token_cost", 0.0))
        mean_latency_ms = float(d.get("mean_latency_ms", 0.0))
        for i in range(n_total):
            acct.record(
                system=f"{system}_baseline",
                n_agents=_N_AGENTS_BASELINE,
                is_correct=(i < n_correct),
                prompt_tokens=0,
                completion_tokens=int(round(mean_tokens)),
                wall_ms=mean_latency_ms,
            )
        n_obs += n_total
        logger.info("Loaded %d obs from %s (system=%s_baseline, N=%d)",
                    n_total, path, system, _N_AGENTS_BASELINE)
    return n_obs


def _load_coordination_jsons(acct: CostAccountant) -> int:
    """results/bench_suite/*/multiagent_benchmark_*.json -- per-mode aggregates."""
    n_obs = 0
    for path in sorted(glob.glob("results/bench_suite/*/multiagent_benchmark_*.json")) + \
            sorted(glob.glob("results/bench_suite/*/*/multiagent_benchmark_*.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        results_by_mode = d.get("results_by_mode", {})
        n_tasks = d.get("metadata", {}).get("n_tasks")
        for mode, metrics in results_by_mode.items():
            n_agents = _N_AGENTS_COMM_MODE.get(mode)
            if n_agents is None or "accuracy" not in metrics:
                continue
            n = n_tasks or 1
            n_correct = int(round(metrics["accuracy"] * n))
            token_cost = float(metrics.get("token_cost", 0.0))
            latency_ms = float(metrics.get("latency_ms", 0.0))
            for i in range(n):
                acct.record(
                    system=mode,
                    n_agents=n_agents,
                    is_correct=(i < n_correct),
                    prompt_tokens=0,
                    completion_tokens=int(round(token_cost)),
                    wall_ms=latency_ms,
                )
            n_obs += n
            logger.info("Loaded %d obs from %s (system=%s, N=%d)", n, path, mode, n_agents)
    return n_obs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/cost_frontier.json")
    args = parser.parse_args()

    acct = CostAccountant()
    n1 = _load_baseline_jsons(acct)
    n2 = _load_coordination_jsons(acct)
    total = n1 + n2
    if total == 0:
        raise RuntimeError(
            "No real cost/accuracy data found under results/baselines/ or "
            "results/bench_suite/ -- nothing to report. Run at least one "
            "baseline or coordination-pipeline eval first."
        )

    report = acct.finalize()
    report.print_frontier()
    out_dict = report.to_dict()
    out_dict["real_n_agents_used"] = sorted({c.n_agents for c in report.cells})
    out_dict["limitations"] = (
        "Source JSONs carry per-run aggregate accuracy/cost only, not "
        "per-sample detail, so each cell's accuracy and mean token/latency "
        "are exact reconstructions of the source files while std_total_tokens "
        "/ std_wall_ms / accuracy_ci_95 reflect this reconstruction, not "
        "measured per-sample spread. CANONICAL_N_VALUES=(4,8,16) in "
        "eval/cost.py's own docstring do not apply here -- this codebase's "
        "agent chain is sequential by design (dev_doc.md §10, "
        "orchestration.parallel_agents was removed rather than wired) and "
        "never runs more than 3 live agent-calls per task; real_n_agents_used "
        "above lists what this repo actually produces."
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    logger.info("%d total observations (%d baseline + %d coordination) -> %s",
                total, n1, n2, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
