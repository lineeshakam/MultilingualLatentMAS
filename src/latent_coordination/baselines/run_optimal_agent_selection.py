"""Optimal-Agent-Selection baseline runner on MGSM / Belebele workloads.

Wraps :class:`~latent_coordination.baselines.optimal_agent_selection.OptimalAgentSelectionBaseline`
and runs it on the same MGSM and Belebele benchmarks used by the other
baseline runners.

Per task, the selector chooses between two real candidate plans under a cost
budget:
    "agent1_only"       — cost 1 (one generation call), utility = Agent 1's
                           own answer-confidence proxy (mean log-probability
                           of its greedy tokens).
    "agent1_plus_agent2"— cost 2 (two generation calls), utility = the same
                           confidence proxy plus a fixed collaboration bonus
                           (this baseline has no oracle for the *true*
                           accuracy gain from a second agent ahead of time,
                           so the bonus is a fixed, documented assumption,
                           not a fabricated per-task oracle score).
This directly exercises cost-constrained subset selection every task, not
just once at startup.

Usage (CLI)
-----------
    python -m latent_coordination.baselines.run_optimal_agent_selection \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --benchmark mgsm --language en \\
        --n 200 --device cuda:0 --budget 2.0 \\
        --output_dir results/baselines/optimal_agent_selection
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
from latent_coordination.eval.correctness import (
    CorrectnessResult,
    CorrectnessScorer,
    load_belebele_tasks,
    load_mgsm_tasks,
    load_mgsm_pro_tasks,
    load_afrimgsm_tasks,
    score_mgsm,
)

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


logger = logging.getLogger(__name__)

_COLLAB_UTILITY_BONUS = 0.3  # documented fixed assumption, see module docstring.


@dataclass
class OptimalAgentSelectionRunConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    benchmark: str = "mgsm"
    language: str = "en"
    split: str = "test"
    n: Optional[int] = 200
    device: str = "cuda:0"
    dtype: str = "float16"
    load_in_8bit: bool = False
    output_dir: str = "results/baselines/optimal_agent_selection"
    seed: int = 42
    max_new_tokens: int = 256
    budget: float = 2.0


@dataclass
class OptimalAgentSelectionRunReport:
    config: Dict
    benchmark: str
    language: str
    n_total: int
    n_correct: int
    accuracy: float
    mean_token_cost: float
    mean_latency_ms: float
    total_wall_clock_s: float
    n_selected_two_agent: int
    n_selected_one_agent: int
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Optimal-Agent-Selection run report saved to %s", path)


def _load_model_and_tokenizer(config: OptimalAgentSelectionRunConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.dtype, torch.float16)
    if dtype == torch.bfloat16:
        raise AssertionError("bf16 is not supported on V100; use float16.")
    load_kwargs: Dict = {
        "torch_dtype": dtype, "trust_remote_code": True, "attn_implementation": "sdpa",
    }
    if config.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs.pop("torch_dtype", None)
        load_kwargs["device_map"] = {"": config.device}
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **load_kwargs)
    if not config.load_in_8bit:
        model = model.to(config.device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _generate_with_confidence(
    model, tokenizer, prompt: str, config: OptimalAgentSelectionRunConfig,
) -> Tuple[str, int, float]:
    """Generate greedily, returning (text, n_new_tokens, mean_token_logprob)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True, return_dict_in_generate=True,
        )
    seq = out.sequences
    n_new = seq.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if out.scores:
        logprobs = []
        for step, logits in enumerate(out.scores):
            token_id = seq[0, inputs["input_ids"].shape[1] + step]
            lp = F.log_softmax(logits[0].float(), dim=-1)[token_id]
            logprobs.append(float(lp.item()))
        mean_logprob = sum(logprobs) / max(len(logprobs), 1)
    else:
        mean_logprob = 0.0
    return text, int(n_new), mean_logprob


def _confidence_utility(mean_logprob: float) -> float:
    """Map a mean token log-probability to a bounded [0, 1] utility."""
    return float(math.exp(max(mean_logprob, -10.0)))


def _select(sel: OptimalAgentSelectionBaseline, config, utility1: float) -> Dict[str, object]:
    candidates = {
        "agent1_only": {"utility": utility1, "cost": 1.0},
        "agent1_plus_agent2": {"utility": utility1 + _COLLAB_UTILITY_BONUS, "cost": 2.0},
    }
    return sel.select_agents(candidates, budget=config.budget)


def run_mgsm(config: OptimalAgentSelectionRunConfig) -> OptimalAgentSelectionRunReport:
    if config.benchmark == "mgsm_pro":
        tasks = load_mgsm_pro_tasks(language=config.language, n=config.n)
    elif config.benchmark == "afrimgsm":
        tasks = load_afrimgsm_tasks(language=config.language, split=config.split, n=config.n)
    else:
        tasks = load_mgsm_tasks(language=config.language, split=config.split, n=config.n)
    logger.info("Loaded %d %s tasks (lang=%s)", len(tasks), config.benchmark, config.language)

    model, tokenizer = _load_model_and_tokenizer(config)
    sel = OptimalAgentSelectionBaseline()

    results: List[CorrectnessResult] = []
    token_costs: List[int] = []
    latencies_ms: List[float] = []
    n_two_agent = 0
    n_one_agent = 0
    t_total = time.perf_counter()

    for task in tasks:
        t0 = time.perf_counter()
        total_tokens = 0
        prompt1 = f"Solve step by step: {task['question']}\nReasoning:"
        text1, n1, logprob1 = _generate_with_confidence(model, tokenizer, prompt1, config)
        total_tokens += n1

        plan = _select(sel, config, _confidence_utility(logprob1))
        use_agent2 = "agent1_plus_agent2" in plan["selected_agents"]

        if use_agent2:
            n_two_agent += 1
            prompt2 = f"{prompt1}\n{text1}\nDouble-check and give the final numeric answer:"
            inputs2 = tokenizer(prompt2, return_tensors="pt", truncation=True, max_length=1024)
            inputs2 = {k: v.to(config.device) for k, v in inputs2.items()}
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs2, max_new_tokens=config.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            n2 = out_ids.shape[1] - inputs2["input_ids"].shape[1]
            text2 = tokenizer.decode(out_ids[0, inputs2["input_ids"].shape[1]:], skip_special_tokens=True)
            total_tokens += int(n2)
            final_text = text2
        else:
            n_one_agent += 1
            final_text = text1

        result = score_mgsm(final_text, float(task["answer"]))
        results.append(result)
        token_costs.append(total_tokens)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        logger.info(
            "OptimalAgentSelection MGSM task %d/%d | lang=%s | 2-agent=%s | running_acc=%.3f",
            len(results), len(tasks), config.language, use_agent2,
            sum(r.is_correct for r in results) / len(results),
        )

    total_wall = time.perf_counter() - t_total
    n_correct = sum(r.is_correct for r in results)
    accuracy = n_correct / max(len(results), 1)
    return OptimalAgentSelectionRunReport(
        config=asdict(config), benchmark=config.benchmark, language=config.language,
        n_total=len(results), n_correct=n_correct, accuracy=accuracy,
        mean_token_cost=sum(token_costs) / max(len(token_costs), 1),
        mean_latency_ms=sum(latencies_ms) / max(len(latencies_ms), 1),
        total_wall_clock_s=total_wall,
        n_selected_two_agent=n_two_agent, n_selected_one_agent=n_one_agent,
    )


def run_belebele(config: OptimalAgentSelectionRunConfig) -> OptimalAgentSelectionRunReport:
    tasks = load_belebele_tasks(language=config.language, split=config.split, n=config.n)
    logger.info("Loaded %d Belebele tasks (lang=%s)", len(tasks), config.language)

    model, tokenizer = _load_model_and_tokenizer(config)
    sel = OptimalAgentSelectionBaseline()
    scorer = CorrectnessScorer(model=model, tokenizer=tokenizer, device=config.device)

    results: List[CorrectnessResult] = []
    token_costs: List[int] = []
    latencies_ms: List[float] = []
    n_two_agent = 0
    n_one_agent = 0
    t_total = time.perf_counter()

    for task in tasks:
        t0 = time.perf_counter()
        total_tokens = 0
        prompt1 = f"Passage: {task['passage']}\nQuestion: {task['question']}\nAnalysis:"
        text1, n1, logprob1 = _generate_with_confidence(model, tokenizer, prompt1, config)
        total_tokens += n1

        plan = _select(sel, config, _confidence_utility(logprob1))
        use_agent2 = "agent1_plus_agent2" in plan["selected_agents"]

        answer_prompt = (
            f"Passage: {task['passage']}\nQuestion: {task['question']}\n"
            f"Analysis: {text1 if use_agent2 else ''}\nAnswer:"
        )
        result = scorer.score_multiple_choice(
            prompt=answer_prompt, choices=task["choices"], gold_idx=task["correct_idx"],
            benchmark="belebele",
        )
        total_tokens += sum(
            len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in task["choices"]
        )
        if use_agent2:
            n_two_agent += 1
        else:
            n_one_agent += 1

        results.append(result)
        token_costs.append(total_tokens)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        logger.info(
            "OptimalAgentSelection Belebele task %d/%d | lang=%s | 2-agent=%s | running_acc=%.3f",
            len(results), len(tasks), config.language, use_agent2,
            sum(r.is_correct for r in results) / len(results),
        )

    total_wall = time.perf_counter() - t_total
    n_correct = sum(r.is_correct for r in results)
    accuracy = n_correct / max(len(results), 1)
    return OptimalAgentSelectionRunReport(
        config=asdict(config), benchmark="belebele", language=config.language,
        n_total=len(results), n_correct=n_correct, accuracy=accuracy,
        mean_token_cost=sum(token_costs) / max(len(token_costs), 1),
        mean_latency_ms=sum(latencies_ms) / max(len(latencies_ms), 1),
        total_wall_clock_s=total_wall,
        n_selected_two_agent=n_two_agent, n_selected_one_agent=n_one_agent,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Optimal-Agent-Selection baseline runner")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--benchmark", choices=["mgsm", "mgsm_pro", "afrimgsm", "belebele"], default="mgsm")
    parser.add_argument("--language", default="en")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--output_dir", default="results/baselines/optimal_agent_selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--budget", type=float, default=2.0)
    args = parser.parse_args()

    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    cfg = OptimalAgentSelectionRunConfig(
        model_id=args.model_id, benchmark=args.benchmark, language=args.language,
        split=args.split, n=args.n, device=args.device, dtype=args.dtype,
        load_in_8bit=args.load_in_8bit, output_dir=args.output_dir, seed=args.seed,
        max_new_tokens=args.max_new_tokens, budget=args.budget,
    )

    report = run_mgsm(cfg) if args.benchmark in ("mgsm", "mgsm_pro", "afrimgsm") else run_belebele(cfg)

    out_dir = Path(args.output_dir)
    ts = report.timestamp_utc
    out_path = out_dir / f"optimal_agent_selection_{args.benchmark}_{args.language}_{ts}.json"
    report.save_json(out_path)
    print(f"accuracy={report.accuracy:.4f}  n={report.n_total}  tokens/task={report.mean_token_cost:.1f}")
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
