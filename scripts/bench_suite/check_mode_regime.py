#!/usr/bin/env python
"""Print cached mode-result files whose code_regime predates the router fix.

Usage: check_mode_regime.py <_results_dir> [mode ...]

For each requested mode (default: token_based_mas, latent_based_mas_ours),
scans <_results_dir>/*mode__<mode>.pt for CheckpointManager.cache_result
payloads (see src/shared/checkpointing.py) and prints one path per line for
every file whose stamped code_regime.router != "prototype-seeded" (a stamp
missing from the payload, e.g. pre-3f75cfb runs, also counts as stale).

If NO cache file exists at all for a requested mode (or the results dir is
absent), prints the sentinel line "MISSING::<mode>" -- previously this case
printed nothing, which was indistinguishable from "cache present and
fixed-regime" and caused requeue_router_fix.sh to mark het_mgsm's never-run
latent mode as done (2026-07-11). Consumers must treat any output as "rerun
needed" and only `mv` lines that are real files (the sentinel is not one).

Prints nothing, and exits 0, only when every requested mode has a
fixed-regime cache present -- this is a read-only probe used by
scripts/bench_suite/requeue_router_fix.sh to decide what to invalidate.
"""
import sys
from pathlib import Path

import torch

FIXED_REGIME = "prototype-seeded"
DEFAULT_MODES = ("token_based_mas", "latent_based_mas_ours")


def main() -> None:
    results_dir = Path(sys.argv[1])
    modes = sys.argv[2:] or DEFAULT_MODES
    for mode in modes:
        matches = sorted(results_dir.glob(f"*mode__{mode}.pt")) if results_dir.is_dir() else []
        if not matches:
            print(f"MISSING::{mode}")
            continue
        for path in matches:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                regime = payload.get("obj", {}).get("code_regime", {})
            except Exception:
                regime = {}
            if regime.get("router") != FIXED_REGIME:
                print(path)


if __name__ == "__main__":
    main()
