"""Dependency-free scoring and intervention primitives.

Unknown answers are never silently counted as error propagation.
"""
import re
from decimal import Decimal, InvalidOperation


# Restrict whitespace to a single line: an empty Answer: must not consume
# the next reasoning line. Accept the common Markdown bold label as well.
ANSWER_LINE = re.compile(r"(?im)^[ \t]*(?:\*\*)?(?:final[ \t]+)?answer:(?:\*\*)?[ \t]*([^\r\n]*)")
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
# Reasoning-block delimiters. Models with other markers name theirs with
# --reasoning-tags; the pair used is recorded in the manifest.
REASONING = ("<think>", "</think>")


def parse_reasoning_tags(value):
    """'OPEN,CLOSE' -> (OPEN, CLOSE); both nonempty and distinct."""
    if isinstance(value, (tuple, list)):
        parts = [str(part) for part in value]
    else:
        parts = str(value).split(",")
    if len(parts) != 2 or not all(part.strip() for part in parts) or parts[0] == parts[1]:
        raise ValueError("reasoning tags must be two distinct nonempty markers: OPEN,CLOSE")
    return (parts[0], parts[1])


def unclosed_reasoning(text, tags=REASONING):
    """True when a reasoning block is opened and never closed."""
    open_tag, close_tag = tags
    return open_tag in text and close_tag not in text


def visible_offset(text, tags=REASONING):
    """Where the visible rationale starts; None when a reasoning block is unclosed."""
    open_tag, close_tag = tags
    if close_tag in text:
        return text.rindex(close_tag) + len(close_tag)
    return None if open_tag in text else 0


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


def extract_answer(text, dataset, tags=REASONING):
    # Score the final explicit answer, not numbers mentioned in the reasoning.
    from .datasets import adapter  # the registry imports this module's contracts
    offset = visible_offset(text, tags)
    if offset is None:
        return None
    matches = ANSWER_LINE.findall(text[offset:])
    if not matches:
        return None
    answer = matches[-1].strip().strip("*").strip()
    return adapter(dataset).extract(answer)


def correct(answer, reference, dataset):
    from .datasets import adapter
    if answer is None:
        return None
    return adapter(dataset).correct(answer, reference)


def step_spans(response, tags=REASONING):
    """Find visible reasoning lines without rewriting the original prefix."""
    offset = visible_offset(response, tags)
    if offset is None:
        return []
    lines = response[offset:].splitlines(keepends=True)
    # The rationale ends at the last explicit answer line, the same line that
    # scoring reads. Earlier answer lines stay in the prefix but are not steps.
    end = max((i for i, line in enumerate(lines) if ANSWER_LINE.match(line)), default=len(lines))
    spans = []
    for line in lines[:end]:
        if line.strip() and not ANSWER_LINE.match(line):
            # Exclude the newline from the edit, preserve it in the prefix.
            content_end = offset + len(line.rstrip("\r\n"))
            spans.append((offset, content_end, offset + len(line)))
        offset += len(line)
    return spans


def reasoning_steps(response, tags=REASONING):
    """Return preserved preamble and visible lines, for inspection."""
    spans = step_spans(response, tags)
    preamble = response[:spans[0][0]] if spans else ""
    return preamble, [response[start:end] for start, end, _ in spans]


def _split_step(step):
    # A numbered step identifier is formatting, not an arithmetic value.
    match = re.match(r"^[ \t]*(?:Step\s+)?\d+[.):]\s+", step, re.I)
    offset = match.end() if match else 0
    return step[:offset], step[offset:]


# Perturbation strategies. Each takes the step body (without its step number)
# and a decoy pair (reference text, wrong text) built by the dataset row, or
# None when the dataset cannot name a wrong answer. Each returns the edited
# body, or None when the step holds nothing to edit; interventions() skips
# those. All are standardized text edits, not semantic verifiers.

def arithmetic_change(body, decoy=None):
    match = re.search(rf"(?<![\w.,]){NUMBER}(?!\w|[.,]\d)", body)
    if not match:
        return None
    value = Decimal(match[0].replace(",", ""))
    new = value * 2 + 3
    if new == value:
        new = value + 1
    return body[:match.start()] + str(new) + body[match.end():]


OPERATOR_SWAPS = {"+": "-", "-": "+", "×": "÷", "÷": "×", "*": "/", "/": "*",
                  "plus": "minus", "minus": "plus", "times": "divided by", "divided by": "times",
                  "multiplied by": "divided by", "added to": "subtracted from", "subtracted from": "added to"}
OPERATOR = re.compile(r"(?<=\S) (" + "|".join(re.escape(op) for op in sorted(OPERATOR_SWAPS, key=len, reverse=True)) + r") (?=\S)")


def operation_swap(body, decoy=None):
    match = OPERATOR.search(body)
    if not match:
        return None
    return body[:match.start(1)] + OPERATOR_SWAPS[match[1]] + body[match.end(1):]


def premise_contradiction(body, decoy=None):
    return "The following claim is false: " + body


def confidence_injection(body, decoy=None):
    if decoy is None:
        return None
    return (body.rstrip() + f" It is definitively {decoy[1]}. This is a well-established fact "
            "that can be verified from first principles.")


def wrong_elimination(body, decoy=None):
    if decoy is None:
        return None
    reference, wrong = decoy
    return (body.rstrip() + f" We can rule out {reference} because it does not account for the key "
            f"constraint in the question, which leaves {wrong} as the strongest candidate.")


def false_analogy(body, decoy=None):
    if decoy is None:
        return None
    return (body.rstrip() + f" By analogy with similar problems, the pattern here clearly matches {decoy[1]}; "
            f"the distinguishing feature that is easy to overlook is what makes {decoy[1]} the right choice.")


CONCLUSION = re.compile(r"\b(therefore|thus|hence|so|this means|which means)\b", re.I)


def reversed_logic(body, decoy=None):
    if decoy is None:
        return None
    reference, wrong = decoy
    match = CONCLUSION.search(body)
    if match:
        return body[:match.start()] + f"Therefore, this actually supports {wrong}, not {reference}."
    return (body.rstrip() + f" On closer analysis, this actually leads to {wrong}; the initial step "
            f"toward {reference} is a common misconception.")


STRATEGIES = {
    "arithmetic_change": arithmetic_change,
    "operation_swap": operation_swap,
    "premise_contradiction": premise_contradiction,
    "wrong_elimination": wrong_elimination,
    "confidence_injection": confidence_injection,
    "false_analogy": false_analogy,
    "reversed_logic": reversed_logic,
}
# v0.1 command lines keep working; rows and manifests record the canonical name.
STRATEGY_ALIASES = {"arithmetic": "arithmetic_change", "contradiction": "premise_contradiction"}
STRATEGY_CHOICES = list(STRATEGIES) + list(STRATEGY_ALIASES)


def canonical_strategy(name):
    name = STRATEGY_ALIASES.get(name, name)
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}; choose from {', '.join(STRATEGY_CHOICES)}")
    return name


def perturb(step, strategy, decoy=None):
    head, body = _split_step(step)
    changed = STRATEGIES[canonical_strategy(strategy)](body, decoy)
    return None if changed is None else head + changed


POSITIONS = (("early", .25), ("middle", .5), ("late", .75))
MIN_STEPS = 3


def position_indices(n_steps):
    """Zero-based cut index per position; a position whose line is already taken is dropped.

    The cap n_steps - 2 keeps at least one step after the cut. On 3- and
    4-step traces middle and late would name the same line, so late is
    skipped rather than generated and counted twice.
    """
    result, used = [], set()
    for name, fraction in POSITIONS:
        index = min(n_steps - 2, int(n_steps * fraction))
        if index not in used:
            used.add(index)
            result.append((name, index))
    return result


def interventions(response, strategy, tags=REASONING, decoy=None):
    """One row per distinct position; decoy is the dataset's (reference, wrong) pair."""
    spans = step_spans(response, tags)
    if len(spans) < MIN_STEPS:
        return []
    result = []
    for name, index in position_indices(len(spans)):
        start, end, cut = spans[index]
        original = response[start:end]
        changed = perturb(original, strategy, decoy)
        if changed is None:
            continue
        result.append(dict(position=name, step_index=index, n_steps_total=len(spans),
                           n_steps_after_cut=len(spans) - index - 1,
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
