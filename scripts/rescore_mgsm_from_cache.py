#!/usr/bin/env python3
"""Rescore cached MGSM answers with the current extract_mgsm_answer.

The 2026-07-11 extractor fix (\\boxed{} support; fallback no longer dies on
bare punctuation) means runs launched before it were scored under a stricter
extractor than runs launched after. This script rescores each mode's cached
answers uniformly under the CURRENT extractor so cross-mode comparisons are
apples-to-apples, and writes a sidecar summary per config — it never modifies
the caches themselves.

Works on ``::partial`` chunk caches (which persist answers + scored tasks,
including each task's gold answer). Completed-mode caches store only
task_details without golds; rerun those modes or rescore before completion.

Usage:
    python scripts/rescore_mgsm_from_cache.py [--config het_mgsm hom_mgsm]

Author: Himon Thakur
License: Apache 2.0
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from latent_coordination.eval.correctness import score_mgsm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def rescore_partial(pt_path: str) -> dict | None:
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    obj = d.get("obj", d)
    answers = obj.get("answers")
    scored_tasks = obj.get("scored_tasks")
    if not answers or not scored_tasks:
        return None
    n_seen = n_old_style = n_new = n_gold = 0
    per_lang: dict = {}
    for ans, task in zip(answers, scored_tasks):
        meta = getattr(task, "metadata", None) or {}
        if meta.get("benchmark") != "mgsm" or "gold_answer" not in meta:
            continue
        n_gold += 1
        text = getattr(ans, "output_text", None) or str(ans)
        gold = float(meta["gold_answer"])
        res = score_mgsm(text, gold)
        lang = getattr(task, "target_language", None) or str(
            getattr(task, "task_id", "")).split("_")[1:2]
        slot = per_lang.setdefault(str(lang), [0, 0])
        slot[1] += 1
        slot[0] += int(res.is_correct)
        n_new += int(res.is_correct)
        n_seen += 1
    if not n_seen:
        return None
    return {
        "cache": Path(pt_path).name,
        "n_scored": n_seen,
        "n_correct_current_extractor": n_new,
        "accuracy_current_extractor": round(n_new / n_seen, 4),
        "per_language": {k: {"n_correct": v[0], "n": v[1],
                             "accuracy": round(v[0] / v[1], 4)}
                         for k, v in sorted(per_lang.items())},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", nargs="+", default=["het_mgsm", "hom_mgsm"])
    args = p.parse_args()

    for cfg in args.config:
        out = {"config": cfg,
               "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
               "note": ("accuracy under the 2026-07-11 extract_mgsm_answer; "
                        "compare against the mode's live-scored accuracy to "
                        "quantify extractor-version skew"),
               "modes": []}
        pattern = f".cache/checkpoints/bench_suite/{cfg}/coordination/_results/*__partial.pt"
        for pt in sorted(glob.glob(pattern)):
            logger.info("rescoring %s", pt)
            r = rescore_partial(pt)
            if r:
                r["mode"] = pt.split("__mode__")[1].replace("__partial.pt", "")
                out["modes"].append(r)
                logger.info("[%s] %s: acc=%.4f over %d tasks", cfg, r["mode"],
                            r["accuracy_current_extractor"], r["n_scored"])
        if out["modes"]:
            dest = Path(f"results/bench_suite/{cfg}/mgsm_rescore_summary.json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(out, indent=2))
            logger.info("summary -> %s", dest)
        else:
            logger.info("[%s] no rescorable partial caches", cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
