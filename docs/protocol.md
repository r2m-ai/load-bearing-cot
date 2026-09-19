# Continuation-based causal testing · load-bearing-v0.1

This specification describes `evaluate.py`. The [paper pipeline](paper-reproduction.md) remains the reference for the published experiments. The new benchmark exposes the same intervention logic with a smaller, explicit configuration surface; its scores should not be substituted for the paper's numbers.

## A worked example

![A single-step perturbation followed by silent bypass, self-correction or error propagation.](assets/continuation_protocol.svg)

The example illustrates the paper's three-way taxonomy. The reusable evaluator scores answer correctness and leaves the A/B distinction to further trace-level analysis.

## Estimand

For a sampled question, generate a baseline response. Retain it only if the extracted answer is correct and the visible rationale has at least three nonempty lines before the answer. For each supported intervention position, replace one reasoning line, truncate, and continue. Score only the newly generated continuation, so an answer embedded in the prefix cannot be mistaken for a recovered answer.

Let `S` contain interventions with a parseable perturbed answer, and let `P` contain interventions with parseable answers in both the perturbed and control conditions:

```text
error_propagation_rate = wrong perturbed answers in S / |S|
answer_coverage = |S| / number of generated interventions
paired_error_rate_difference = mean_P[wrong(perturbed) - wrong(control)]
```

`control_error_rate` uses the same paired set `P`. A negative difference is allowed. Undefined quantities are `null`; zero scored continuations produces `status: insufficient_data`. The binary error proxy does not establish uptake of the injected content. No A/B judge is used.

Report baseline accuracy, baseline parse failures, eligible-question count, intervention count and answer coverage alongside the metric. Baseline accuracy counts an unparseable baseline as unsuccessful; perturbation scores instead exclude unparseable answers and expose their count. Missingness may be model-dependent, so coverage is essential to interpretation.

Three positions share a question, and short traces can map multiple positions to the same line. Overall scores weight generated interventions equally, not questions equally. Do not treat these observations as independent trials; uncertainty estimates should resample at question level. This entry point does not yet compute confidence intervals or a leaderboard ranking.

## Versioned design choices

| Component | v0.1 behavior |
|---|---|
| Input sampling | Seeded shuffle; 100 questions by default, `--all` for all rows |
| GSM8K | `openai/gsm8k`, main/test; keep reference solutions with at least three calculation-marker lines |
| MMLU | `cais/mmlu`, all/test; four choices with a letter reference |
| BBH | `lukaemon/bbh`, test; all available configurations or explicit `--task`; task load failures abort |
| Prompt | Explicit step-by-step reasoning with one step per line and `Answer:` on its own line |
| Decoding | Greedy, 4,096 new tokens by default in each condition |
| Steps | Nonempty lines before the first explicit final-answer line; original whitespace is preserved in both prefixes |
| Reasoning tags | With `</think>`, retain the entire thinking preamble and intervene only on the visible rationale after it; an unclosed `<think>` is ineligible |
| Position | `min(n_steps - 2, floor(n_steps × fraction))`, zero-based; fractions 0.25, 0.50, 0.75 |
| Arithmetic edit | First numeric value after any step number becomes `2n + 3`; use `n + 1` if unchanged; skip lines without a number |
| Contradiction edit | Prefix the step's content with `The following claim is false:` |
| Default strategy | Arithmetic on GSM8K; contradiction on MMLU and BBH; explicitly overridable |
| Continuation | Render the original user prompt once with the model chat template, append the assistant prefix, then generate raw token-prompt completions |
| Control | Repeat continuation with the unedited prefix at the same position |
| Scoring | Last explicit `Answer:` line (also accepts a bold Markdown label) in the generated suffix; numeric equality for GSM8K, strict A–D parsing for MMLU, case-insensitive exact match for BBH (single-letter option parentheses are normalized; bracket sequences are preserved) |

The contradiction intervention is a standardized textual negation, not a semantic verifier; inspect traces to check its suitability for a task. BBH uses exact normalized string matching, without the historical prefix-matching heuristic. Formatting variation may lower measured accuracy or coverage. Do not silently change parsers between model runs.

For reasoning-tag models, the retained thinking preamble may already contain a solution. This measures dependence on the **post-thinking visible rationale**, not dependence on the internal thinking trace. Newer models with different reasoning delimiters need an explicit adapter and protocol review; accepting a model ID does not validate that model's reasoning format.

The backend uses [vLLM token prompts](https://docs.vllm.ai/en/latest/api/vllm/inputs/) to continue an assistant prefix without introducing a follow-up user turn. Models with no chat template or incompatible architectures fail rather than silently using a different protocol.

## Reproducibility and artifacts

Supply `--model-revision` and `--dataset-revision` with commit hashes when publishing a run. Default upstream revisions can change. The manifest records these arguments, package versions, dataset fingerprints, the sampled IDs and content hash, the chat template, and the repository commit/dirty state. A manifest without pinned revisions is useful provenance, not an exact environment lock.

`baselines.jsonl` records every sampled question, including incorrect and unparseable baselines. `interventions.jsonl` records every attempted valid edit and both continuation outcomes. Each row carries the base question ID for grouped analysis. Baseline records are appended before continuation generation, and intervention records after paired generations complete; interrupted partial questions are not resumable automatically. Keyboard interruption is recorded as `interrupted` in the manifest; both JSONL files exist even when no interventions qualify. Choose a new output directory for a retry.

The shared dependency file has lower bounds. GPU/library combinations and model architectures must be validated before reporting new results. There are no new GPU-validated benchmark results shipped with this entry point.

## Extend to a new model or dataset

A vLLM-compatible causal model with a supported chat template can be supplied as a Hugging Face ID or local path. Aliases in `load_bearing/backend.py` are conveniences, not evidence of validation. The paper's three model families and the Qwen example are kept distinct in the README.

For local evaluation data, use JSONL with unique IDs:

```json
{"id":"example-1","question":"A box has 4 rows of 6 items. How many items?","reference_answer":"24"}
```

```bash
python evaluate.py --model gemma-2-9b-it --dataset gsm8k \
  --input-file my_questions.jsonl --all
```

`--dataset` selects the answer contract for local files. MMLU-format rows additionally require four `choices` and a letter `reference_answer`; BBH-format rows use a textual reference and may include `task`. Local rows are not filtered by reference-solution markers. Keep dataset provenance with the file when sharing results.

For a new answer type, add a dataset adapter, parser fixtures and intervention checks before claiming support. Agent trajectories require a separate intervention unit, task-success measure and continuation backend; they are outside v0.1.

## Difference from the published experiments

The paper uses seven perturbation strategies, historical model-specific prompts and selection rules, rule-plus-judge classification, human annotation, hidden-state probes and steering. v0.1 uses two explicit edit types, strict parsing, a uniform visible-line rule, and paired unedited-prefix controls. It provides a reusable behavioral measurement, not a one-command numerical reproduction of all paper tables.

Useful comparisons hold strategy, position policy, parser, token budget and dataset sample fixed. Correct-baseline conditioning still selects different examples for different models. For a matched cross-model comparison, additionally report results on shared eligible questions. Task difficulty is model-relative; this is not a universal scale of intrinsic problem difficulty or mechanistic faithfulness.
