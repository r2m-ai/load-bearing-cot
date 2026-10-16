"""Dataset registry; all rows share id/question/reference_answer fields.

Adding a dataset means adding one row to DATASETS. The CLI choices, the
default strategy, the Hugging Face source, the row conversion, the input
validation, the prompt wording and the answer contract all read from it.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import random
from typing import Callable, Optional

from .protocol import correct_number, correct_text, extract_choice, extract_number, extract_text


@dataclass(frozen=True)
class Dataset:
    source: str                      # Hugging Face dataset ID
    configs: Optional[tuple]         # None: discover configurations at load time (one task each)
    split: str
    convert: Callable                # upstream row -> question/reference_answer fields, or None to skip
    validate: Callable               # input row -> None, or raise ValueError
    question: Callable               # input row -> question text for the prompt
    answer_type: str                 # wording of the "Answer: <...>" instruction
    extract: Callable                # answer line -> normalized answer or None
    correct: Callable                # (normalized answer, reference) -> bool
    default_strategy: str


def _convert_gsm8k(row):
    # Match the paper's reference-solution multistep filter.
    if sum("<<" in line for line in row["answer"].splitlines()) < 3:
        return None
    return dict(question=row["question"], reference_answer=row["answer"].split("####")[-1].strip())


def _convert_mmlu(row):
    return dict(question=row["question"], choices=row["choices"], reference_answer="ABCD"[row["answer"]])


def _convert_bbh(row):
    return dict(question=row["input"], reference_answer=str(row["target"]))


def _validate_number(row):
    try:
        valid = Decimal(row["reference_answer"].replace(",", "")).is_finite()
    except InvalidOperation:
        valid = False
    if not valid:
        raise ValueError("GSM8K reference_answer must be a finite number")


def _validate_choice(row):
    if (not isinstance(row.get("choices"), list) or len(row["choices"]) != 4
            or not all(isinstance(c, str) and c.strip() for c in row["choices"])
            or row["reference_answer"] not in ("A", "B", "C", "D")):
        raise ValueError("MMLU rows need four choices and an A/B/C/D reference_answer")


def _validate_text(row):
    return None


def _question_with_choices(row):
    return row["question"] + "\n" + "\n".join(f"{letter}) {choice}" for letter, choice in zip("ABCD", row["choices"]))


def _question(row):
    return row["question"]


DATASETS = {
    "gsm8k": Dataset(source="openai/gsm8k", configs=("main",), split="test",
                     convert=_convert_gsm8k, validate=_validate_number, question=_question,
                     answer_type="number", extract=extract_number, correct=correct_number,
                     default_strategy="arithmetic"),
    "mmlu": Dataset(source="cais/mmlu", configs=("all",), split="test",
                    convert=_convert_mmlu, validate=_validate_choice, question=_question_with_choices,
                    answer_type="letter", extract=extract_choice, correct=correct_text,
                    default_strategy="contradiction"),
    "bbh": Dataset(source="lukaemon/bbh", configs=None, split="test",
                   convert=_convert_bbh, validate=_validate_text, question=_question,
                   answer_type="your answer", extract=extract_text, correct=correct_text,
                   default_strategy="contradiction"),
}


def adapter(name):
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset: {name}; registered: {', '.join(DATASETS)}") from None


def load_examples(name, limit, seed, task=None, revision=None, input_file=None):
    dataset = adapter(name)
    if input_file:
        with open(input_file) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        provenance = {"input_file": str(input_file)}
    else:
        from datasets import load_dataset, get_dataset_config_names
        if dataset.configs is not None:
            configs = list(dataset.configs)
        else:
            configs = [task] if task else sorted(get_dataset_config_names(dataset.source, revision=revision))
        rows, fingerprints = [], {}
        for config in configs:
            data = load_dataset(dataset.source, config, split=dataset.split, revision=revision)
            fingerprints[config] = data._fingerprint
            for index, row in enumerate(data):
                fields = dataset.convert(row)
                if fields is None:
                    continue
                rows.append(dict({"id": f"{name}/{config}/{index}", "task": row.get("subject", config)}, **fields))
        provenance = {"source": dataset.source, "split": dataset.split, "configs": configs,
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
        dataset.validate(row)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows, provenance


def format_prompt(row, name):
    dataset = adapter(name)
    return ("Solve the following problem step by step. Put each reasoning step on a separate line. "
            f"Give the final answer on its own line as: Answer: <{dataset.answer_type}>\n\n"
            f"Question: {dataset.question(row)}\n\nSolution:")
