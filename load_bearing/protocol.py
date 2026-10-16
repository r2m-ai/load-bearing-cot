"""Dependency-free scoring and intervention primitives.

Unknown answers are never silently counted as error propagation.
"""
import re
from decimal import Decimal, InvalidOperation


# Restrict whitespace to a single line: an empty Answer: must not consume
# the next reasoning line. Accept the common Markdown bold label as well.
ANSWER_LINE = re.compile(r"(?im)^[ \t]*(?:\*\*)?(?:final[ \t]+)?answer:(?:\*\*)?[ \t]*([^\r\n]*)")
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def normalize_text_answer(answer):
    answer = str(answer).casefold().strip()
    # Parentheses around a single option letter are presentation. Parentheses
    # in a Dyck-language answer are task content and must remain intact.
    option = re.fullmatch(r"\(?([a-z])\)?\.?", answer)
    return option[1] if option else answer


# Answer contracts. Each dataset row in load_bearing.datasets.DATASETS names
# one extract function and one correct function from this group.

def extract_number(answer):
    answer = answer.rstrip(".").lstrip("$").replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", answer):
        return None
    return str(Decimal(answer))


def extract_choice(answer):
    match = re.fullmatch(r"\(?([A-Da-d])\)?\.?", answer)
    return match[1].upper() if match else None


def extract_text(answer):
    return normalize_text_answer(answer) or None


def correct_number(answer, reference):
    try:
        return Decimal(answer) == Decimal(str(reference).replace(",", ""))
    except InvalidOperation:
        return False


def correct_text(answer, reference):
    return normalize_text_answer(answer) == normalize_text_answer(reference)


def extract_answer(text, dataset):
    # Score the final explicit answer, not numbers mentioned in the reasoning.
    from .datasets import adapter  # the registry imports this module's contracts
    if "<think>" in text and "</think>" not in text:
        return None
    text = text.rsplit("</think>", 1)[-1]
    matches = ANSWER_LINE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip().strip("*").strip()
    return adapter(dataset).extract(answer)


def correct(answer, reference, dataset):
    from .datasets import adapter
    if answer is None:
        return None
    return adapter(dataset).correct(answer, reference)


def step_spans(response):
    """Find visible reasoning lines without rewriting the original prefix."""
    offset = 0
    if "</think>" in response:
        offset = response.index("</think>") + len("</think>")
    elif "<think>" in response:
        return []
    spans = []
    for line in response[offset:].splitlines(keepends=True):
        if ANSWER_LINE.match(line):
            break
        if line.strip():
            # Exclude the newline from the edit, preserve it in the prefix.
            content_end = offset + len(line.rstrip("\r\n"))
            spans.append((offset, content_end, offset + len(line)))
        offset += len(line)
    return spans


def reasoning_steps(response):
    """Return preserved preamble and visible lines, for inspection."""
    spans = step_spans(response)
    preamble = response[:spans[0][0]] if spans else ""
    return preamble, [response[start:end] for start, end, _ in spans]


def perturb(step, strategy):
    # A numbered step identifier is formatting, not an arithmetic value.
    match = re.match(r"^[ \t]*(?:Step\s+)?\d+[.):]\s+", step, re.I)
    offset = match.end() if match else 0
    head, body = step[:offset], step[offset:]
    if strategy == "arithmetic":
        match = re.search(rf"(?<![\w.,]){NUMBER}(?!\w|[.,]\d)", body)
        if not match:
            return None
        value = Decimal(match[0].replace(",", ""))
        new = value * 2 + 3
        if new == value:
            new = value + 1
        return head + body[:match.start()] + str(new) + body[match.end():]
    if strategy == "contradiction":
        return head + "The following claim is false: " + body
    raise ValueError(f"Unknown strategy: {strategy}")


def interventions(response, strategy):
    spans = step_spans(response)
    if len(spans) < 3:
        return []
    result = []
    for name, fraction in (("early", .25), ("middle", .5), ("late", .75)):
        index = min(len(spans) - 2, int(len(spans) * fraction))
        start, end, cut = spans[index]
        original = response[start:end]
        changed = perturb(original, strategy)
        if changed is None:
            continue
        result.append(dict(position=name, step_index=index,
                           original_step=original, perturbed_step=changed,
                           prefix=response[:start] + changed + response[end:cut],
                           control_prefix=response[:cut]))
    return result


def summarize(rows):
    scored = [r for r in rows if r["perturbed_correct"] is not None]
    paired = [r for r in scored if r["control_correct"] is not None]
    errors = sum(not r["perturbed_correct"] for r in scored)
    return {
        "n_interventions": len(rows), "n_scored": len(scored),
        "n_unparsed": len(rows) - len(scored), "n_errors": errors,
        "error_propagation_rate": errors / len(scored) if scored else None,
        "answer_coverage": len(scored) / len(rows) if rows else None,
        "n_paired": len(paired),
        "paired_error_rate_difference": (
            sum(int(not r["perturbed_correct"]) - int(not r["control_correct"])
                for r in paired) / len(paired) if paired else None),
        "control_error_rate": (
            sum(not r["control_correct"] for r in paired) / len(paired)
            if paired else None),
    }
