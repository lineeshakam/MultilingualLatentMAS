#!/usr/bin/env python
"""Run XCOPA with single/text/latent planner-solver-critic prompting.

XCOPA is a two-choice causal commonsense benchmark. This runner mirrors the
MGSM planner/solver/critic comparison but scores option selection instead of a
numeric answer:

1. single_solver
2. text_planner_solver_critic
3. latent_planner_solver_critic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from latent_coordination.agents.base_agent import AgentConfig, AgentTask, BaseAgent


XCOPA_LANGS = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]


def load_xcopa_dataset(lang: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets library required: pip install datasets") from exc

    errors = []
    for dataset_id in ["xcopa", "cambridgeltl/xcopa"]:
        try:
            return load_dataset(dataset_id, lang, split=split)
        except Exception as exc:  # pragma: no cover - depends on HF dataset aliases.
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("Could not load XCOPA. Tried:\n" + "\n".join(errors))


@dataclass
class XcopaItem:
    task_id: str
    lang: str
    idx: int
    premise: str
    question: str
    choice1: str
    choice2: str
    label: int


def load_items(languages: Iterable[str], split: str, n: int, start_idx: int = 0) -> List[XcopaItem]:
    items: List[XcopaItem] = []
    for lang in languages:
        ds = load_xcopa_dataset(lang, split)
        rows = list(ds)
        selected = rows[start_idx:] if n < 0 else rows[start_idx:start_idx + n]
        for idx, row in enumerate(selected, start=start_idx):
            items.append(XcopaItem(
                task_id=f"xcopa_{lang}_{idx}",
                lang=lang,
                idx=idx,
                premise=str(row["premise"]),
                question=str(row["question"]),
                choice1=str(row["choice1"]),
                choice2=str(row["choice2"]),
                label=int(row["label"]),
            ))
    return items


def format_problem(item: XcopaItem) -> str:
    relation = "cause" if item.question.lower() == "cause" else "effect"
    return (
        f"Premise: {item.premise}\n"
        f"Question: Which option is the more plausible {relation}?\n"
        f"1. {item.choice1}\n"
        f"2. {item.choice2}\n"
        "Choose exactly one option."
    )


def build_prompt(role: str, item: XcopaItem, context: str = "") -> str:
    problem = format_problem(item)
    if role == "planner":
        return (
            "You are a causal commonsense planning agent. Identify the causal "
            "relation and list the key evidence needed to choose option 1 or 2. "
            "Do not answer yet.\n\n"
            f"{problem}\n\nPlan:"
        )
    if role == "solver":
        return (
            "You are a careful causal commonsense solver. Pick the more plausible "
            "choice. End with exactly: Answer: 1 or Answer: 2.\n\n"
            f"{problem}\n"
            f"{context}\n\nSolution:"
        )
    if role == "critic":
        return (
            "You are a causal commonsense critic. Check whether the proposed choice "
            "really follows as the requested cause/effect. If wrong, correct it. "
            "End with exactly: Corrected answer: 1 or Corrected answer: 2.\n\n"
            f"{problem}\n"
            f"{context}\n\nCritique:"
        )
    if role == "reviser":
        return (
            "You are the final causal commonsense solver. Use the available critique "
            "or latent feedback and output only the final option. End with exactly: "
            "Answer: 1 or Answer: 2.\n\n"
            f"{problem}\n"
            f"{context}\n\nFinal answer:"
        )
    raise ValueError(f"unknown role: {role}")


def extract_choice(text: str) -> Optional[int]:
    if not isinstance(text, str) or not text.strip():
        return None
    patterns = [
        r"(?:corrected\s+answer|final\s+answer|answer)\s*[:：]?\s*(?:option\s*)?([12])\b",
        r"\boption\s*([12])\b",
        r"\bchoice\s*([12])\b",
        r"^\s*([12])\s*[\).:-]?",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return int(m.group(1)) - 1
    # Conservative fallback: if the final non-empty line is just "1" or "2".
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        m = re.fullmatch(r"(?:answer\s*[:：]?\s*)?([12])[\).]?", lines[-1], flags=re.IGNORECASE)
        if m:
            return int(m.group(1)) - 1
    return None


class PromptRoleAgent(BaseAgent):
    def __init__(self, config: AgentConfig, prompt_role: str, capture_layer: int) -> None:
        super().__init__(config)
        self.prompt_role = prompt_role
        self.capture_layer = capture_layer

    def process(self, task: AgentTask):
        raise NotImplementedError("Use run_role() in this standalone runner.")

    def run_role(
        self,
        item: XcopaItem,
        context: str = "",
        latent_state: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
    ):
        prompt = build_prompt(self.prompt_role, item, context)
        t0 = time.perf_counter()
        text, latent = self.generate_and_capture(
            prompt,
            latent_state=latent_state,
            injection_layer=self.capture_layer,
            capture_layer=self.capture_layer,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            do_sample=False,
        )
        return text.strip(), latent, (time.perf_counter() - t0) * 1000.0


def make_agents(args) -> Dict[str, PromptRoleAgent]:
    base_cfg = AgentConfig(
        agent_id="agent_solver",
        model_id=args.model_name,
        role="reasoning",
        device=args.device,
        hidden_dim=args.hidden_dim,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        latent_transfer_layer=args.latent_transfer_layer,
        max_time_s=args.max_time_s,
    )
    solver = PromptRoleAgent(base_cfg, "solver", args.latent_transfer_layer)
    solver._ensure_model_loaded()
    agents = {"solver": solver}
    for role in ["planner", "critic", "reviser"]:
        cfg = AgentConfig(
            agent_id=f"agent_{role}",
            model_id=args.model_name,
            role="reasoning",
            device=args.device,
            hidden_dim=args.hidden_dim,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            latent_transfer_layer=args.latent_transfer_layer,
            max_time_s=args.max_time_s,
        )
        agent = PromptRoleAgent(cfg, role, args.latent_transfer_layer)
        agent._model = solver._model
        agent._tokenizer = solver._tokenizer
        agent._is_loaded = True
        agents[role] = agent
    return agents


def score_output(text: str, label: int) -> Dict:
    pred = extract_choice(text)
    return {"prediction": None if pred is None else pred + 1, "correct": pred == label}


def run_single_solver(item: XcopaItem, agents: Dict[str, PromptRoleAgent], args) -> Dict:
    text, _, elapsed = agents["solver"].run_role(item, max_new_tokens=args.max_new_tokens)
    scored = score_output(text, item.label)
    return {
        **scored,
        "elapsed_ms": elapsed,
        "planner_text": "",
        "solver_text": text,
        "critic_text": "",
        "final_text": text,
        "output_text": text,
    }


def run_text_psc(item: XcopaItem, agents: Dict[str, PromptRoleAgent], args) -> Dict:
    plan, _, t_plan = agents["planner"].run_role(item, max_new_tokens=args.plan_tokens)
    sol, _, t_sol = agents["solver"].run_role(
        item, f"\nPlanner output:\n{plan}", max_new_tokens=args.max_new_tokens
    )
    crit, _, t_crit = agents["critic"].run_role(
        item, f"\nPlanner output:\n{plan}\n\nProposed solution:\n{sol}", max_new_tokens=args.critic_tokens
    )
    final, _, t_final = agents["reviser"].run_role(
        item, f"\nPlanner output:\n{plan}\n\nInitial solution:\n{sol}\n\nCritique:\n{crit}", max_new_tokens=args.max_new_tokens
    )
    scored = score_output(final, item.label)
    return {
        **scored,
        "elapsed_ms": t_plan + t_sol + t_crit + t_final,
        "planner_text": plan,
        "solver_text": sol,
        "critic_text": crit,
        "final_text": final,
        "output_text": final,
    }


def run_latent_psc(item: XcopaItem, agents: Dict[str, PromptRoleAgent], args) -> Dict:
    plan, z_plan, t_plan = agents["planner"].run_role(item, max_new_tokens=args.plan_tokens)
    sol, z_sol, t_sol = agents["solver"].run_role(
        item,
        "\nA latent planning signal is available. Use it to choose the option.",
        latent_state=z_plan,
        max_new_tokens=args.max_new_tokens,
    )
    crit, z_crit, t_crit = agents["critic"].run_role(
        item,
        "\nA latent solution signal is available. Check the option.",
        latent_state=z_sol,
        max_new_tokens=args.critic_tokens,
    )
    final, _, t_final = agents["reviser"].run_role(
        item,
        "\nA latent critique signal is available. Produce the final option.",
        latent_state=z_crit,
        max_new_tokens=args.max_new_tokens,
    )
    scored = score_output(final, item.label)
    return {
        **scored,
        "elapsed_ms": t_plan + t_sol + t_crit + t_final,
        "planner_text": plan,
        "solver_text": sol,
        "critic_text": crit,
        "final_text": final,
        "output_text": final,
    }


def normalize_modes(value: str) -> List[str]:
    if value == "all":
        return ["single_solver", "text_planner_solver_critic", "latent_planner_solver_critic"]
    return [m.strip() for m in value.split(",") if m.strip()]


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "examples.csv", index=False)
    summary = df.groupby("mode")["correct"].agg(accuracy="mean", correct="sum", total="count").reset_index()
    summary.to_csv(out_dir / "summary_by_mode.csv", index=False)
    lang_summary = df.groupby(["mode", "lang"])["correct"].agg(accuracy="mean", correct="sum", total="count").reset_index()
    lang_summary.to_csv(out_dir / "summary_by_mode_lang.csv", index=False)
    wide = df.pivot_table(index=["task_id", "lang", "idx"], columns="mode", values="correct", aggfunc="first").reset_index()
    wide.to_csv(out_dir / "overlap_wide.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "summary_by_mode": summary.to_dict(orient="records"),
        "summary_by_mode_lang": lang_summary.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="CohereLabs/aya-expanse-8b")
    ap.add_argument("--languages", default=",".join(XCOPA_LANGS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--modes", default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--hidden_dim", type=int, default=4096)
    ap.add_argument("--load_in_8bit", action="store_true", default=True)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--plan_tokens", type=int, default=96)
    ap.add_argument("--critic_tokens", type=int, default=128)
    ap.add_argument("--latent_transfer_layer", type=int, default=-4)
    ap.add_argument("--max_time_s", type=float, default=120.0)
    ap.add_argument("--out_dir", default="results/xcopa_planner_solver_critic")
    ap.add_argument("--run_name", default="xcopa_aya_planner_solver_critic")
    ap.add_argument("--checkpoint_every", type=int, default=10)
    args = ap.parse_args()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    modes = normalize_modes(args.modes)
    items = load_items(languages, args.split, args.max_examples, args.start_idx)
    agents = make_agents(args)
    out_dir = Path(args.out_dir) / args.run_name

    rows: List[Dict] = []
    for n, item in enumerate(items, start=1):
        print(f"=== {item.lang} idx={item.idx} ===", flush=True)
        for mode in modes:
            if mode == "single_solver":
                result = run_single_solver(item, agents, args)
            elif mode == "text_planner_solver_critic":
                result = run_text_psc(item, agents, args)
            elif mode == "latent_planner_solver_critic":
                result = run_latent_psc(item, agents, args)
            else:
                raise ValueError(f"unknown mode: {mode}")
            row = {
                "mode": mode,
                "task_id": item.task_id,
                "lang": item.lang,
                "idx": item.idx,
                "premise": item.premise,
                "question": item.question,
                "choice1": item.choice1,
                "choice2": item.choice2,
                "gold": item.label + 1,
                **result,
            }
            print(f"  {mode}: correct={row['correct']} pred={row['prediction']} gold={row['gold']}", flush=True)
            rows.append(row)
        if args.checkpoint_every > 0 and n % args.checkpoint_every == 0:
            write_outputs(rows, out_dir)
            print(f"  [checkpoint] wrote {len(rows)} rows to {out_dir}", flush=True)

    write_outputs(rows, out_dir)
    print("[OK] wrote", out_dir, flush=True)
    print(pd.read_csv(out_dir / "summary_by_mode.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
