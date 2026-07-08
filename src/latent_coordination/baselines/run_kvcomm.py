"""KVComm baseline runner on MGSM / Belebele workloads.

Wraps :class:`~latent_coordination.baselines.kvcomm.KVCommBaseline` and runs
it on the same MGSM and Belebele benchmarks used by the other baseline
runners, so results are directly comparable on the accuracy-vs-token-cost
frontier.

Simplification note: :meth:`KVCommBaseline.fuse` operates on real
``(key, value)`` cache tensors, but this two-agent text-in/text-out CLI
chain (matching ``run_thoughtcomm.py``'s structure) does not do a live,
per-layer forward-hook splice into ``model.generate()``'s KV cache — that
would require custom generation-loop plumbing beyond this pass's scope.
Instead, Agent 1's last-layer hidden state is treated as a single-token
pseudo (key, value) pair, fused via the real ``KVCommBaseline.fuse()`` call,
and the fused vector is injected into Agent 2 as a soft prefix via
``latent_prefix.py`` (same injection mechanism ``run_thoughtcomm.py`` uses).
The fusion math is real and exercised end-to-end; the "online, live KV
cache" delivery mechanism is not.

Usage (CLI)
-----------
    python -m latent_coordination.baselines.run_kvcomm \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --benchmark mgsm --language en \\
        --n 200 --device cuda:0 \\
        --output_dir results/baselines/kvcomm
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from latent_coordination.baselines.kvcomm import KVCommBaseline
from latent_coordination.baselines.latent_prefix import (
    build_latent_prefix,
    generate_with_latent_prefix,
)
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


@dataclass
class KVCommRunConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    benchmark: str = "mgsm"          # "mgsm" | "mgsm_pro" | "afrimgsm" | "belebele"
    language: str = "en"
    split: str = "test"
    n: Optional[int] = 200
    device: str = "cuda:0"
    dtype: str = "float16"
    load_in_8bit: bool = False
    output_dir: str = "results/baselines/kvcomm"
    seed: int = 42
    max_new_tokens: int = 256


@dataclass
class KVCommRunReport:
    config: Dict
    benchmark: str
    language: str
    n_total: int
    n_correct: int
    accuracy: float
    mean_token_cost: float
    mean_latency_ms: float
    total_wall_clock_s: float
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
        logger.info("KVComm run report saved to %s", path)


def _load_model_and_tokenizer(config: KVCommRunConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    dtype_map = {"float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(config.dtype, torch.float16)
    if dtype == torch.bfloat16:
        raise AssertionError("bf16 is not supported on V100; use float16.")
    load_kwargs: Dict = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
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


def _generate_text(model, tokenizer, prompt: str, config: KVCommRunConfig) -> Tuple[str, int]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    n_new = out_ids.shape[1] - inputs["input_ids"].shape[1]
    text = tokenizer.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, int(n_new)


def _extract_last_hidden(model, tokenizer, text: str, device: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0].mean(dim=0)  # (D,)


def _fuse_as_pseudo_kv(kv: KVCommBaseline, hidden1: torch.Tensor) -> torch.Tensor:
    """Treat a single hidden vector as a 1-token pseudo (K, V) pair and fuse it."""
    k = v = hidden1.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    fused_k, fused_v = kv.fuse("agent1", "agent2", k, v, k, v, layer_idx=0)
    return (fused_k[0, 0] + fused_v[0, 0]) / 2.0


def run_mgsm(config: KVCommRunConfig) -> KVCommRunReport:
    if config.benchmark == "mgsm_pro":
        tasks = load_mgsm_pro_tasks(language=config.language, n=config.n)
    elif config.benchmark == "afrimgsm":
        tasks = load_afrimgsm_tasks(language=config.language, split=config.split, n=config.n)
    else:
        tasks = load_mgsm_tasks(language=config.language, split=config.split, n=config.n)
    logger.info("Loaded %d %s tasks (lang=%s)", len(tasks), config.benchmark, config.language)

    model, tokenizer = _load_model_and_tokenizer(config)
    hidden_dim = model.config.hidden_size
    kv = KVCommBaseline(device=config.device)
    kv.register_agent("agent1", num_heads=1, head_dim=hidden_dim)
    kv.register_agent("agent2", num_heads=1, head_dim=hidden_dim)

    results: List[CorrectnessResult] = []
    token_costs: List[int] = []
    latencies_ms: List[float] = []
    t_total = time.perf_counter()

    for task in tasks:
        t0 = time.perf_counter()
        total_tokens = 0

        prompt1 = f"Solve step by step: {task['question']}\nReasoning:"
        text1, n1 = _generate_text(model, tokenizer, prompt1, config)
        total_tokens += n1

        hidden1 = _extract_last_hidden(model, tokenizer, text1, config.device)
        with torch.no_grad():
            fused = _fuse_as_pseudo_kv(kv, hidden1)

        prompt2 = f"{prompt1}\n{text1}\nFinal numeric answer:"
        text2, n2 = generate_with_latent_prefix(
            model, tokenizer, prompt2, fused, config.device, config.max_new_tokens,
        )
        total_tokens += n2

        result = score_mgsm(text2, float(task["answer"]))
        results.append(result)
        token_costs.append(total_tokens)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        logger.info(
            "KVComm MGSM task %d/%d | lang=%s | task_s=%.1f | running_acc=%.3f",
            len(results), len(tasks), config.language,
            latencies_ms[-1] / 1000, sum(r.is_correct for r in results) / len(results),
        )

    total_wall = time.perf_counter() - t_total
    n_correct = sum(r.is_correct for r in results)
    accuracy = n_correct / max(len(results), 1)
    return KVCommRunReport(
        config=asdict(config), benchmark=config.benchmark, language=config.language,
        n_total=len(results), n_correct=n_correct, accuracy=accuracy,
        mean_token_cost=sum(token_costs) / max(len(token_costs), 1),
        mean_latency_ms=sum(latencies_ms) / max(len(latencies_ms), 1),
        total_wall_clock_s=total_wall,
    )


def run_belebele(config: KVCommRunConfig) -> KVCommRunReport:
    tasks = load_belebele_tasks(language=config.language, split=config.split, n=config.n)
    logger.info("Loaded %d Belebele tasks (lang=%s)", len(tasks), config.language)

    model, tokenizer = _load_model_and_tokenizer(config)
    hidden_dim = model.config.hidden_size
    kv = KVCommBaseline(device=config.device)
    kv.register_agent("agent1", num_heads=1, head_dim=hidden_dim)
    kv.register_agent("agent2", num_heads=1, head_dim=hidden_dim)
    scorer = CorrectnessScorer(model=model, tokenizer=tokenizer, device=config.device)

    results: List[CorrectnessResult] = []
    token_costs: List[int] = []
    latencies_ms: List[float] = []
    t_total = time.perf_counter()

    for task in tasks:
        t0 = time.perf_counter()
        total_tokens = 0

        prompt1 = f"Passage: {task['passage']}\nQuestion: {task['question']}\nAnalysis:"
        text1, n1 = _generate_text(model, tokenizer, prompt1, config)
        total_tokens += n1

        hidden1 = _extract_last_hidden(model, tokenizer, text1, config.device)
        with torch.no_grad():
            fused = _fuse_as_pseudo_kv(kv, hidden1)

        answer_prompt = (
            f"Passage: {task['passage']}\nQuestion: {task['question']}\n"
            f"Analysis: {text1}\nAnswer:"
        )
        with torch.no_grad():
            prompt_ids = tokenizer(
                answer_prompt, return_tensors="pt", truncation=True, max_length=1024,
            )["input_ids"].to(config.device)
            prompt_embeds = model.get_input_embeddings()(prompt_ids)
            prefix_embeds = build_latent_prefix(fused, prompt_embeds)
        result = scorer.score_multiple_choice(
            prompt=answer_prompt, choices=task["choices"], gold_idx=task["correct_idx"],
            benchmark="belebele", prefix_embeds=prefix_embeds,
        )
        total_tokens += sum(
            len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in task["choices"]
        )
        results.append(result)
        token_costs.append(total_tokens)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        logger.info(
            "KVComm Belebele task %d/%d | lang=%s | task_s=%.1f | running_acc=%.3f",
            len(results), len(tasks), config.language,
            latencies_ms[-1] / 1000, sum(r.is_correct for r in results) / len(results),
        )

    total_wall = time.perf_counter() - t_total
    n_correct = sum(r.is_correct for r in results)
    accuracy = n_correct / max(len(results), 1)
    return KVCommRunReport(
        config=asdict(config), benchmark="belebele", language=config.language,
        n_total=len(results), n_correct=n_correct, accuracy=accuracy,
        mean_token_cost=sum(token_costs) / max(len(token_costs), 1),
        mean_latency_ms=sum(latencies_ms) / max(len(latencies_ms), 1),
        total_wall_clock_s=total_wall,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="KVComm baseline runner")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--benchmark", choices=["mgsm", "mgsm_pro", "afrimgsm", "belebele"], default="mgsm")
    parser.add_argument("--language", default="en")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--output_dir", default="results/baselines/kvcomm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    cfg = KVCommRunConfig(
        model_id=args.model_id, benchmark=args.benchmark, language=args.language,
        split=args.split, n=args.n, device=args.device, dtype=args.dtype,
        load_in_8bit=args.load_in_8bit, output_dir=args.output_dir, seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )

    report = run_mgsm(cfg) if args.benchmark in ("mgsm", "mgsm_pro", "afrimgsm") else run_belebele(cfg)

    out_dir = Path(args.output_dir)
    ts = report.timestamp_utc
    out_path = out_dir / f"kvcomm_{args.benchmark}_{args.language}_{ts}.json"
    report.save_json(out_path)
    print(f"accuracy={report.accuracy:.4f}  n={report.n_total}  tokens/task={report.mean_token_cost:.1f}")
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
