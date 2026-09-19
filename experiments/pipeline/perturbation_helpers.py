"""
Shared helpers for v8 experiments. Extracted from experiments/pipeline/perturb_and_continue.py.
"""

import re


def parse_cot_steps(cot_response: str) -> list[str]:
    """Parse CoT into individual reasoning steps."""
    lines = cot_response.strip().split("\n")
    steps = []
    current_step = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_new_step = bool(re.match(r"^(\d+[\.\):]|Step\s+\d+|[-*])\s", stripped))
        if is_new_step and current_step:
            steps.append("\n".join(current_step))
            current_step = [line]
        else:
            current_step.append(line)
    if current_step:
        steps.append("\n".join(current_step))
    return steps


def extract_gsm8k_answer(response: str) -> str | None:
    match = re.search(r"Answer:\s*\$?([\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[\d,]+\.?\d*", response)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_mmlu_answer(response: str) -> str | None:
    match = re.search(r"Answer:\s*\(?([A-Da-d])\)?", response)
    if match:
        return match.group(1).upper()
    match = re.search(r"[Tt]he (?:correct )?answer is\s*\(?([A-Da-d])\)?", response)
    if match:
        return match.group(1).upper()
    match = re.search(r"[Oo]ption\s+([A-Da-d])\b", response)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\)", response)
    if match:
        return match.group(1)
    return None
