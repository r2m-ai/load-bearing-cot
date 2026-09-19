"""Dataset adapters; all rows share id/question/reference_answer fields."""
import json
import random
from decimal import Decimal, InvalidOperation


def load_examples(name, limit, seed, task=None, revision=None, input_file=None):
    if input_file:
        with open(input_file) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        provenance = {"input_file": str(input_file)}
    else:
        from datasets import load_dataset, get_dataset_config_names
        source = {"gsm8k": "openai/gsm8k", "mmlu": "cais/mmlu", "bbh": "lukaemon/bbh"}[name]
        configs = {"gsm8k": ["main"], "mmlu": ["all"]}.get(name)
        if name == "bbh":
            configs = [task] if task else sorted(get_dataset_config_names(source, revision=revision))
        rows, fingerprints = [], {}
        for config in configs:
            data = load_dataset(source, config, split="test", revision=revision)
            fingerprints[config] = data._fingerprint
            for index, row in enumerate(data):
                item = {"id": f"{name}/{config}/{index}", "task": row.get("subject", config)}
                if name == "gsm8k":
                    # Match the paper's reference-solution multistep filter.
                    if sum("<<" in line for line in row["answer"].splitlines()) < 3:
                        continue
                    item.update(question=row["question"], reference_answer=row["answer"].split("####")[-1].strip())
                elif name == "mmlu":
                    item.update(question=row["question"], choices=row["choices"], reference_answer="ABCD"[row["answer"]])
                else:
                    item.update(question=row["input"], reference_answer=str(row["target"]))
                rows.append(item)
        provenance = {"source": source, "split": "test", "configs": configs,
                      "revision": revision, "fingerprints": fingerprints}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not all(key in row for key in ("id", "question", "reference_answer")):
            raise ValueError("Each input row needs id, question, reference_answer")
        if not all(isinstance(row[key], str) and row[key].strip()
                   for key in ("id", "question", "reference_answer")):
            raise ValueError("id, question and reference_answer must be nonempty strings")
        if "task" in row and (not isinstance(row["task"], str) or not row["task"].strip()):
            raise ValueError("task must be a nonempty string")
        if row["id"] in seen:
            raise ValueError(f"Duplicate input id: {row['id']}")
        seen.add(row["id"])
        if name == "mmlu" and (not isinstance(row.get("choices"), list)
                               or len(row["choices"]) != 4
                               or not all(isinstance(c, str) and c.strip() for c in row["choices"])
                               or row["reference_answer"] not in ("A", "B", "C", "D")):
            raise ValueError("MMLU rows need four choices and an A/B/C/D reference_answer")
        if name == "gsm8k":
            try:
                valid = Decimal(row["reference_answer"].replace(",", "")).is_finite()
            except InvalidOperation:
                valid = False
            if not valid:
                raise ValueError("GSM8K reference_answer must be a finite number")
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows, provenance


def format_prompt(row, dataset):
    question = row["question"]
    if dataset == "mmlu":
        question += "\n" + "\n".join(f"{letter}) {choice}" for letter, choice in zip("ABCD", row["choices"]))
    answer_type = {"gsm8k": "number", "mmlu": "letter", "bbh": "your answer"}[dataset]
    return ("Solve the following problem step by step. Put each reasoning step on a separate line. "
            f"Give the final answer on its own line as: Answer: <{answer_type}>\n\nQuestion: {question}\n\nSolution:")
