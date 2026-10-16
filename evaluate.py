#!/usr/bin/env python3
"""Evaluate continuation sensitivity; see docs/protocol.md for the estimand."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from load_bearing import PROTOCOL_VERSION
from load_bearing.backend import MODEL_ALIASES, VLLMBackend, reasoning_tags_for
from load_bearing.datasets import DATASETS, format_prompt, load_examples
from load_bearing.protocol import (POSITIONS, REASONING, STRATEGY_CHOICES, canonical_strategy, correct, extract_answer,
                                   interventions, parse_reasoning_tags, position_indices, step_spans, summarize,
                                   uncertainty, unclosed_reasoning)

# After this many baselines that all open a reasoning block without closing
# it, the delimiters are wrong for this model; stop instead of scoring every
# row as unparseable.
UNCLOSED_REASONING_PATIENCE = 5


def check_reasoning_format(baselines, tags, final=False):
    unclosed = [b for b in baselines if b.get("reasoning_unclosed")]
    if baselines and len(unclosed) == len(baselines) and (final or len(baselines) >= UNCLOSED_REASONING_PATIENCE):
        raise ValueError(f"Every baseline so far ({len(baselines)}) opened {tags[0]!r} without closing it; "
                         f"check --reasoning-tags for this model or raise --max-new-tokens")


def run(examples, dataset, backend, strategy, record, tags=REASONING, seed=42):
    strategy = canonical_strategy(strategy)
    decoy = DATASETS[dataset].decoy
    baselines, rows = [], []
    for index, example in enumerate(examples):
        prompt = format_prompt(example, dataset)
        responses = backend.generate([(prompt, "")])
        if len(responses) != 1:
            raise ValueError("Backend must return exactly one baseline generation")
        response = responses[0]
        answer = extract_answer(response, dataset, tags)
        baseline = dict(example, prompt=prompt, response=response, answer=answer,
                        correct=correct(answer, example["reference_answer"], dataset),
                        reasoning_unclosed=unclosed_reasoning(response, tags))
        edits = interventions(response, strategy, tags, decoy(example)) if baseline["correct"] is True else []
        n_steps = len(step_spans(response, tags))
        baseline.update(n_steps=n_steps, n_interventions=len(edits),
                        n_positions_collapsed=(len(POSITIONS) - len(position_indices(n_steps))
                                               if edits else 0))
        baselines.append(baseline)
        record("baselines", baseline)
        check_reasoning_format(baselines, tags)
        requests = [(prompt, edit[key]) for edit in edits for key in ("prefix", "control_prefix")]
        continuations = backend.generate(requests) if requests else []
        if len(continuations) != len(requests):
            raise ValueError("Backend returned the wrong number of continuations")
        for j, edit in enumerate(edits):
            row = dict(edit, id=example["id"], task=example.get("task", dataset), strategy=strategy,
                       reference_answer=example["reference_answer"])
            for kind, text in zip(("perturbed", "control"), continuations[2*j:2*j+2]):
                extracted = extract_answer(text, dataset, tags)
                row[kind + "_continuation"] = text
                row[kind + "_answer"] = extracted
                row[kind + "_correct"] = correct(extracted, example["reference_answer"], dataset)
            rows.append(row)
            record("interventions", row)
        print(f"[{index + 1}/{len(examples)}] {example['id']}: correct={baseline['correct']}, interventions={len(edits)}", flush=True)
    check_reasoning_format(baselines, tags, final=True)
    report = summarize(rows)
    report.update(uncertainty(rows, seed))
    report.update(n_questions=len(examples),
                  n_baseline_correct=sum(b["correct"] is True for b in baselines),
                  n_baseline_unparsed=sum(b["answer"] is None for b in baselines),
                  n_baseline_unclosed_reasoning=sum(b["reasoning_unclosed"] for b in baselines),
                  n_eligible_questions=sum(b["n_interventions"] > 0 for b in baselines),
                  n_coincident_positions_skipped=sum(b["n_positions_collapsed"] for b in baselines))
    report["baseline_accuracy"] = report["n_baseline_correct"] / len(examples) if examples else None
    report["by_position"] = {p: summarize([r for r in rows if r["position"] == p]) for p, _ in POSITIONS}
    report["by_remaining_steps"] = {k: summarize([r for r in rows if r["n_steps_after_cut"] == k])
                                    for k in sorted({r["n_steps_after_cut"] for r in rows})}
    report["by_task"] = {task: summarize([r for r in rows if r["task"] == task]) for task in sorted({r["task"] for r in rows})}
    report["status"] = "ok" if report["n_scored"] else "insufficient_data"
    return report


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model alias, Hugging Face ID, or local model path")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--max-examples", type=positive, default=100, help="Sampled questions, not correct baselines (default: 100)")
    parser.add_argument("--all", action="store_true", help="Use all dataset rows")
    parser.add_argument("--task", help="Single BBH task (default: all available BBH tasks)")
    parser.add_argument("--input-file", type=Path, help="Local JSONL dataset in the documented schema")
    parser.add_argument("--strategy", choices=STRATEGY_CHOICES, default=None,
                        help="Perturbation (default: the dataset's row in load_bearing/datasets.py); "
                             "arithmetic and contradiction are aliases of arithmetic_change and premise_contradiction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=positive, default=4096)
    parser.add_argument("--reasoning-tags", metavar="OPEN,CLOSE",
                        help="Reasoning-block delimiters (default: per-model alias entry, else <think>,</think>)")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--tensor-parallel-size", type=positive, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.9)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are never overwritten")
    args = parser.parse_args()
    if args.task and args.dataset != "bbh":
        parser.error("--task applies only to BBH")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    args.strategy = canonical_strategy(args.strategy or DATASETS[args.dataset].default_strategy)
    if args.reasoning_tags is not None:
        try:
            parse_reasoning_tags(args.reasoning_tags)
        except ValueError as error:
            parser.error(f"--reasoning-tags: {error}")
    output = args.output_dir or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists():
        parser.error(f"--output-dir already exists: {output}; choose a new directory")
    # exist_ok stays False so a directory created between the check and the call still fails.
    output.mkdir(parents=True, exist_ok=False)
    model_id = MODEL_ALIASES.get(args.model, args.model)
    tags = reasoning_tags_for(model_id, args.reasoning_tags)
    manifest = {"protocol": PROTOCOL_VERSION, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model_id": model_id, "reasoning_tags": list(tags), "python": platform.python_version(),
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
        manifest["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(__file__).parent, text=True))
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    manifest_path = output / "manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    def record(name, row):
        with (output / f"{name}.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_manifest()
    for name in ("baselines", "interventions"):
        (output / f"{name}.jsonl").touch()
    try:
        examples, provenance = load_examples(args.dataset, None if args.all else args.max_examples,
                                             args.seed, args.task, args.dataset_revision, args.input_file)
        if not examples:
            raise ValueError("Dataset has no eligible input rows")
        manifest["dataset"] = provenance
        manifest["sample_sha256"] = hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()
        manifest["sample_ids"] = [row["id"] for row in examples]
        manifest["packages"] = {name: importlib.metadata.version(name) for name in ("vllm", "transformers", "torch", "datasets")}
        save_manifest()
        backend = VLLMBackend(args)
        manifest["chat_template"] = backend.tokenizer.chat_template
        save_manifest()
        summary = run(examples, args.dataset, backend, args.strategy, record, tags, args.seed)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest["status"] = summary["status"]
        print(json.dumps(summary, indent=2))
        print(f"Artifacts: {output}")
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()


if __name__ == "__main__":
    main()
