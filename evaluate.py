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
from load_bearing.backend import MODEL_ALIASES, VLLMBackend
from load_bearing.datasets import format_prompt, load_examples
from load_bearing.protocol import correct, extract_answer, interventions, summarize


def run(examples, dataset, backend, strategy, record):
    baselines, rows = [], []
    for index, example in enumerate(examples):
        prompt = format_prompt(example, dataset)
        responses = backend.generate([(prompt, "")])
        if len(responses) != 1:
            raise ValueError("Backend must return exactly one baseline generation")
        response = responses[0]
        answer = extract_answer(response, dataset)
        baseline = dict(example, prompt=prompt, response=response, answer=answer,
                        correct=correct(answer, example["reference_answer"], dataset))
        edits = interventions(response, strategy) if baseline["correct"] is True else []
        baseline["n_interventions"] = len(edits)
        baselines.append(baseline)
        record("baselines", baseline)
        requests = [(prompt, edit[key]) for edit in edits for key in ("prefix", "control_prefix")]
        continuations = backend.generate(requests) if requests else []
        if len(continuations) != len(requests):
            raise ValueError("Backend returned the wrong number of continuations")
        for j, edit in enumerate(edits):
            row = dict(edit, id=example["id"], task=example.get("task", dataset), strategy=strategy,
                       reference_answer=example["reference_answer"])
            for kind, text in zip(("perturbed", "control"), continuations[2*j:2*j+2]):
                extracted = extract_answer(text, dataset)
                row[kind + "_continuation"] = text
                row[kind + "_answer"] = extracted
                row[kind + "_correct"] = correct(extracted, example["reference_answer"], dataset)
            rows.append(row)
            record("interventions", row)
        print(f"[{index + 1}/{len(examples)}] {example['id']}: correct={baseline['correct']}, interventions={len(edits)}", flush=True)
    report = summarize(rows)
    report.update(n_questions=len(examples),
                  n_baseline_correct=sum(b["correct"] is True for b in baselines),
                  n_baseline_unparsed=sum(b["answer"] is None for b in baselines),
                  n_eligible_questions=sum(b["n_interventions"] > 0 for b in baselines))
    report["baseline_accuracy"] = report["n_baseline_correct"] / len(examples) if examples else None
    report["by_position"] = {p: summarize([r for r in rows if r["position"] == p]) for p in ("early", "middle", "late")}
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
    parser.add_argument("--dataset", required=True, choices=["gsm8k", "mmlu", "bbh"])
    parser.add_argument("--max-examples", type=positive, default=100, help="Sampled questions, not correct baselines (default: 100)")
    parser.add_argument("--all", action="store_true", help="Use all dataset rows")
    parser.add_argument("--task", help="Single BBH task (default: all available BBH tasks)")
    parser.add_argument("--input-file", type=Path, help="Local JSONL dataset in the documented schema")
    parser.add_argument("--strategy", choices=["arithmetic", "contradiction"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=positive, default=4096)
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
    args.strategy = args.strategy or ("arithmetic" if args.dataset == "gsm8k" else "contradiction")
    output = args.output_dir or Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": PROTOCOL_VERSION, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model_id": MODEL_ALIASES.get(args.model, args.model), "python": platform.python_version(),
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
        summary = run(examples, args.dataset, backend, args.strategy, record)
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
