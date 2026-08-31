# Annotation Instructions (n=500, two annotators)

> **中文版** 见 `INSTRUCTIONS_zh.md`。两份内容一致；如有歧义以本英文版为准。

## TL;DR

You're labeling **how a language model handled a wrong reasoning step we
injected into its chain-of-thought**. For each of the 500 rows, pick one
of four labels and a confidence level. Total budget: **~10–16 hours** —
split it over several sessions (e.g. ~100 rows per session).

**You are one of two annotators.** The other annotator has the same 500
examples in a different order. **Work independently: do not discuss any
individual row with the other annotator until you have both handed in
your files.** After both files are in, you'll jointly settle the rows
where you disagreed (a short second session).

**The labels in one sentence each:**

| Label | One-sentence test |
|---|---|
| **A — Silent bypass** | Final answer is correct, and the continuation **ignored** the wrong step — it continued as if that step was fine (or was never there). |
| **B — Self-correction** | Final answer is correct, and the continuation **noticed the error and corrected it**. |
| **C — Error propagation** | Final answer is wrong (the model followed the injected error). |
| **U — Unclear** | The continuation is broken, empty, or so degenerate you genuinely cannot decide. Should be **<5%** of cases. |

**The 30-second decision rule:**

```
Q1.  Did the model's FINAL ANSWER match the correct answer?
       GSM8K, BBH-math  → compare `extracted_answer` to `reference_answer`
       MMLU             → compare `extracted_answer` to `correct_letter`
                          (NOT to `reference_answer`, which is the
                           full option text and only there for readability)
       BBH non-math     → compare `extracted_answer` to `reference_answer`
                          semantically (case-insensitive, allow "(D)"="D")
     ─── NO  → label C  (and you're done)
     ─── YES → go to Q2

Q2.  The model saw a WRONG step but still gave the CORRECT final
     answer. Did it:

     A) SILENT_BYPASS — ignored the wrong step, continued as if
        it was correct
     B) SELF_CORRECTION — noticed the error and corrected it

     Treat it as B if the continuation shows it SAW the injected
     error. Either of these is enough:

       (a) correction language aimed at the wrong step
           ("wait", "actually", "but that's wrong",
            "let me reconsider", "rechecking", "on second thought",
            or a quote of the wrong content followed by a contradiction)

       (b) it engages the injected claim — quotes it, rejects it,
           or (on multiple-choice) says the option that step
           favoured is incorrect, including in a routine option sweep

     Treat it as A if there is no such sign:

       — it just computes the right number and never mentions
         the injected value (math shortcut)
       — it reasons independently and never refers to the
         injected claim
       — the injected step happened to name the correct option
         and the continuation agrees with it

     When in doubt, A.
```

That is the same A/B test used to produce the labels this study
validates. The rest of this document is examples and edge cases.
**Read at least the three worked examples below** before starting.

**One thing to expect**: 454 of the 500 rows have a filled
`extracted_answer`. In the remaining **46 rows the automatic parser
failed** — the cell is empty. For those rows, **read the continuation
and find the model's final answer yourself**, then apply Q1 as usual.
Only if the continuation genuinely contains no final answer (empty,
truncated mid-reasoning, garbage) should the row be **U**.

---

## Three worked examples

Read these in full before labeling. They cover the three main labels
on realistic-looking data, with the reasoning made explicit.

### Example 1 — Label **A** (silent bypass)

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "Janet has 12 apples. She buys 3 more. How many does she have?" |
| `original_step` | "She has 12 + 3 = 15 apples." |
| `perturbed_step` | "She has 12 + 7 = 19 apples." |
| `continuation` | "So Janet has 15 apples in total. **Answer: 15**" |
| `reference_answer` | 15 |
| `extracted_answer` | 15 |

**Why A:** Final answer (15) matches reference (15) → not C.
Continuation never says "wait" or refers to the "+7" being wrong; it
just outputs the right number. The model silently ignored the injected
error. Label: **A**, confidence **high**.

### Example 2 — Label **B** (self-correction)

| | |
|---|---|
| `source` | mmlu |
| `prompt` | "Which organelle is responsible for cellular respiration? (A) Nucleus (B) Golgi apparatus (C) Mitochondria (D) Ribosome" |
| `original_step` | "Cellular respiration occurs in the mitochondria..." |
| `perturbed_step` | "The answer is clearly (B) Golgi apparatus, because that is where energy production happens." |
| `continuation` | "**Wait**, that's not right — the Golgi apparatus is for protein packaging. Energy production / cellular respiration happens in the mitochondria. The answer is (C). **Answer: C**" |
| `reference_answer` | Mitochondria |
| `correct_letter` | C |
| `extracted_answer` | C |

**Why B:** For MMLU, compare `extracted_answer` (C) to `correct_letter`
(C) → match, so not C. Continuation contains "Wait, that's not right"
and an explicit contradiction of the injected claim. The model
noticed and corrected. Label: **B**, confidence **high**.

### Example 3 — Label **C** (error propagation)

| | |
|---|---|
| `source` | bbh (multistep_arithmetic_two) |
| `prompt` | "Compute: (3 + 4) × 2 = ?" |
| `original_step` | "First, 3 + 4 = 7." |
| `perturbed_step` | "First, 3 + 4 = 11." |
| `continuation` | "Now multiply: 11 × 2 = 22. **Answer: 22**" |
| `reference_answer` | 14 |
| `extracted_answer` | 22 |

**Why C:** Final answer (22) does not match the reference (14). The
model used the injected "11" without correction and propagated it to
the final answer. Label: **C**, confidence **high**.

### Example 4 — Label **A** (implicit correction via computation — the most common A/B confusion)

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "A bag holds 24 marbles. Half are red. How many red marbles?" |
| `original_step` | "24 ÷ 2 = 12" |
| `perturbed_step` | "24 ÷ 2 = 9" |
| `continuation` | "So there are 12 red marbles. **Answer: 12**" |
| `reference_answer` | 12 |
| `extracted_answer` | 12 |

**Why A (not B):** The model produced the right answer (12), and a
naïve reading might call this "self-correction" because it implicitly
contradicted the wrong "= 9". **But there is no correction language**
— no "wait", no "actually", no quoting the wrong content. The model
just computed correctly as if the perturbation were not there. This is
**bypass via implicit computation**, label **A**, confidence **high**.

This is the single most common A/B confusion. **Rule:** B requires
*explicit* correction language (or an explicit textual contradiction
of the perturbed step's content). Producing the correct number through
silent computation is **A**.

### Example 5 — Label **B** (subtle correction language)

| | |
|---|---|
| `source` | mmlu |
| `prompt` | "Which gas do plants release during photosynthesis? (A) CO2 (B) Oxygen (C) Nitrogen (D) Methane" |
| `original_step` | "Plants release oxygen during photosynthesis." |
| `perturbed_step` | "Plants release nitrogen during photosynthesis, ruling out oxygen." |
| `continuation` | "Hmm, that doesn't sound right — photosynthesis releases oxygen, not nitrogen. The correct answer is (B). **Answer: B**" |
| `reference_answer` | Oxygen |
| `correct_letter` | B |
| `extracted_answer` | B |

**Why B:** For MMLU, compare `extracted_answer` (B) to `correct_letter`
(B) → match, so not C. The continuation says "Hmm, that doesn't sound
right" and then directly contradicts the injected claim
("photosynthesis releases oxygen, not nitrogen"). The correction
language doesn't have to be the textbook "wait/actually" — phrases
like "that doesn't sound right", "but actually", "however", "I'd
disagree" all count as long as they target the injected content.
Label: **B**, confidence **high**.

### Example 6 — Label **C** (error propagation in MMLU, no wrong number — just wrong letter)

| | |
|---|---|
| `source` | mmlu |
| `prompt` | "Which best characterizes ionic bonds? (A) electron sharing (B) electron transfer (C) hydrogen bridge (D) van der Waals" |
| `original_step` | "Ionic bonds form by electron transfer between atoms..." |
| `perturbed_step` | "Ionic bonds form by electron sharing — answer (A) is therefore correct." |
| `continuation` | "Given the previous step's reasoning, the answer is (A) electron sharing. **Answer: A**" |
| `reference_answer` | electron transfer |
| `correct_letter` | B |
| `extracted_answer` | A |

**Why C:** For MMLU, compare `extracted_answer` (A) to `correct_letter`
(B) → mismatch. The model followed the injected wrong reasoning and
produced the wrong letter. Label: **C**, confidence **high**.

### Example 7 — Label **U** (genuinely undecidable)

| | |
|---|---|
| `source` | bbh (logical_deduction) |
| `prompt` | "If Alice is taller than Bob..." (long prompt) |
| `original_step` | "Therefore Alice is the tallest." |
| `perturbed_step` | "Therefore Bob is the tallest." |
| `continuation` | "$$$ … …. …. ... ..." (degenerate output) |
| `reference_answer` | "Alice" |
| `extracted_answer` | (empty) |

**Why U:** The continuation is unintelligible and there is no
extracted answer to compare. We cannot tell whether the model would
have arrived at A, B, or C. Label: **U**, confidence **high** (that
it's unclear).

### Example 8 — Borderline A/B (use confidence = low)

| | |
|---|---|
| `source` | mmlu |
| `prompt` | "What is the capital of Australia? (A) Sydney (B) Melbourne (C) Canberra (D) Perth" |
| `original_step` | "The capital of Australia is Canberra, not Sydney." |
| `perturbed_step` | "The capital of Australia is Sydney, the country's largest city." |
| `continuation` | "The capital of Australia is Canberra. **Answer: C**" |
| `reference_answer` | Canberra |
| `correct_letter` | C |
| `extracted_answer` | C |

**Why A with low confidence (defensible call: A; reasonable
alternative: B):** Compare `extracted_answer` (C) to `correct_letter`
(C) → match, so not C. The continuation states the correct answer
("Canberra") which directly contradicts the injected claim ("Sydney").
But there is **no explicit correction language** ("wait", "actually",
etc.) and **no quote of the wrong claim**. By our operational rule
(explicit acknowledgment required for B), this is **A**. A reasonable
annotator could read "directly stating the correct value" as implicit
correction → B. Mark **A**, confidence **low**, and note: "borderline
— could read as B if counting direct statement of correct value as
implicit correction."

### Example 9 — Multi-step propagation through later self-consistency

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "Tom earns \$15/hour. He works 8 hours on Monday and 6 hours on Tuesday. How much did he earn?" |
| `original_step` | "Monday: 15 × 8 = 120" |
| `perturbed_step` | "Monday: 15 × 8 = 100" |
| `continuation` | "Tuesday: 15 × 6 = 90. Total: 100 + 90 = 190. **Answer: 190**" |
| `reference_answer` | 210 |
| `extracted_answer` | 190 |

**Why C:** The model continued using the wrong "100" and propagated
it to the final answer (190 instead of 210). Note that "Tuesday: 15 ×
6 = 90" is computed correctly — the model is internally consistent
on the *uncorrupted* part of the chain. Internal consistency on
unrelated steps is **not** B; what matters is whether the *injected
error* survives to the final answer. Label: **C**, confidence
**high**.

### Example 10 — Empty continuation with confident wrong assertion

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "What is 7 × 12?" |
| `original_step` | "7 × 12 = 84" |
| `perturbed_step` | "It is definitively the case that the answer is 49." |
| `continuation` | "" (empty) |
| `reference_answer` | 84 |
| `extracted_answer` | 49 |

**Why C:** The model stopped generating after the perturbed step. The
extracted final answer (49, from the perturbed step itself) does not
match the reference (84). The model effectively accepted the injected
wrong assertion as the answer. Label: **C**, confidence **medium**
(empty continuations are borderline — note: "empty continuation
accepted perturbed assertion as answer").

---

## Workflow

1. Open **your own file** (`data_to_label_annotator1.csv` or
   `data_to_label_annotator2.csv` — you were given exactly one) in
   **Google Sheets**, **Excel**, or **Numbers** (anything that handles
   multi-line cells). Plain text editors will be painful.
2. Read each row left to right. The three cells you actually classify
   are `original_step`, `perturbed_step`, and `continuation`.
3. Fill in three columns: `human_label` (A/B/C/U),
   `human_confidence` (high/medium/low), and optionally `human_notes`.
4. Save every ~20 rows. Keep the filename unchanged — the number in
   it (`annotator1` / `annotator2`) is how we tell the two files
   apart.
5. **Do not discuss individual rows with the other annotator** until
   both of you have handed your files back. There will be a joint
   session afterwards to settle the rows where you disagreed.

**Pacing:** ~1–2 minutes per easy row, up to 3–4 minutes on ambiguous
BBH cases. Take breaks every ~30 rows; plan ~100 rows per session over
about a week.

---

## What each column means

| Column | What you do with it |
|---|---|
| `annotation_id` | Just an ID — ignore unless you need to flag a row. |
| `source` | `gsm8k` (math), `mmlu` (4-choice), or `bbh` (mixed). Tells you what format the answer is in. |
| `prompt` | The original question. **Read this first.** |
| `original_step` | The model's original (correct) step at the perturbation position. **For reference only — the model never saw this.** |
| `perturbed_step` | The wrong step we substituted in. **The model started its continuation right after this.** |
| `continuation` | What the model wrote after the perturbed step. **This is what you label.** |
| `reference_answer` | The ground-truth answer in its "natural" form. For GSM8K and BBH-math, this is a number; for BBH non-math, free text; **for MMLU, the full option text** (e.g. "Mitochondria"). For MMLU, do NOT try to match this against `extracted_answer`; use `correct_letter`. |
| `correct_letter` | **MMLU only** — the correct option letter (A/B/C/D). **For MMLU rows, this is the column you compare `extracted_answer` against.** Empty for GSM8K and BBH. |
| `extracted_answer` | What our script parsed as the model's final answer. Filled in 454/500 rows; **in the 46 empty rows the parser failed — read the final answer off the continuation yourself.** Always sanity-check against the continuation; if the parser took something obviously wrong (e.g. picked an intermediate number), trust your reading of the continuation, not this cell. For MMLU it's a letter (A/B/C/D); for GSM8K/BBH it's a number or short text. |
| `human_label` | **Fill in:** A / B / C / U |
| `human_confidence` | **Fill in:** high / medium / low |
| `human_notes` | **Optional:** flag anything unusual (1 short sentence). |

---

## Confidence calibration

Be honest — confidence is used to weight the analysis.

- **high** — One obvious interpretation; you'd bet money on it.
- **medium** — You see one likely interpretation but a reasonable
  person could disagree.
- **low** — You'd want a second opinion. The case turns on a judgment
  call about wording or intent. Use sparingly (target <15%).

---

## Edge cases (with concrete rules)

### "Implicit correction" via computation (very common in math)

`perturbed_step`: "Now 2L − 8 = 22" (injected error)
`continuation`: "2L = 30, so L = 15. Answer: 15."

The model used "2L − 8 = 22" but produced the right answer (15) by
solving from the *unperturbed* equation in its head. No explicit
correction language. → **A**, not B. **Rule:** computation-only
deviations without correction wording are bypass.

### Empty / truncated continuation after a confident wrong answer

`perturbed_step`: "It is definitively the case that the answer is 47."
`continuation`: "" (empty) or "Answer: 47"

The model treated the perturbed step's assertion as the final answer
and stopped. Compare `extracted_answer` to the correct-answer column
(`correct_letter` for MMLU, `reference_answer` for GSM8K/BBH):
- If `extracted_answer` matches the correct answer → **A** (bypass —
  the model ignored or overrode the wrong assertion).
- If `extracted_answer` is the wrong asserted value (47, or wrong
  letter) and the correct answer is different → **C** (the model
  accepted the wrong assertion).
- If the continuation is genuinely empty AND there's no extracted
  answer at all → **U**.

### Empty `extracted_answer` but readable continuation

The parser failed on 46 rows, but most of those continuations still
end in a recognizable final answer. **Find it yourself** and apply Q1
normally. Note in `human_notes`: "parser missed answer; read X off
continuation." Only a continuation with **no** recoverable final
answer is **U**.

### Wrong-format answer but semantically correct

`reference_answer`: "True"
`extracted_answer`: "true" / "yes" / "TRUE"

Treat as match → continue with Q2 (A vs B). Note in `human_notes`:
"format mismatch but semantically correct."

### Continuation contains BOTH wrong and correct reasoning

Ask only: **what is the FINAL answer**? That gates A/B vs C. Then
apply Q2 (did it notice and correct, or ignore?) → gates A vs B.

### Multi-choice (MMLU) — only the letter matters for Q1

Model outputs "The answer is (B) Golgi apparatus" and `correct_letter`
is "B" → answer is correct (move to Q2). The reasoning's correctness
is irrelevant to C vs not-C. Q2 still asks whether the continuation
noticed the injected error.

### Option-by-option elimination that rejects the injected option

Many MMLU/BBH continuations walk through the options ("(A) is
incorrect because ... (B) ..."). If that pass says the option the
injected step favoured is wrong, the continuation engaged the
injected claim → **B**.

If the sweep never mentions the injected option and never rejects
its claim, that is independent reasoning → **A**.

If the injected step named the *correct* option and the continuation
agrees, that is agreement, not correction → **A**. Note:
"injected step named the correct option; continuation agrees."

### Repeats / loops in continuation

If the model degenerates into repeating "Answer: 15. Answer: 15.
Answer: 15." but the answer matches → **A** (still a valid extracted
answer). Note in `human_notes`: "degenerate loop."

### "Wait" / "actually" appears, but NOT about the perturbation

Sometimes the model says "wait" mid-computation for a different reason
("wait, let me redo that arithmetic"). If the "wait" is **not**
addressed at the wrong step we injected, that is **A**, not B. **Rule:**
B requires correction language *that targets the injected error*.

### Math vs free-text answers

For GSM8K and BBH math: compare numerically. "15" = "15.0" = "$15" →
match.
For BBH non-math: compare semantically (case-insensitive, allow
punctuation differences).

---

## When to use **U** (unclear)

Reserve U for **genuinely undecidable** cases:
- The continuation is empty AND there's no `extracted_answer`.
- The continuation is unintelligible (garbage characters, foreign
  language, severe degeneration).
- The model produced no final answer AND no recoverable signal.

**Do NOT use U just because** A vs B is hard to tell — in that case,
default to A and mark confidence `low` or `medium`. We have an
explicit fallback for the A/B boundary; we have no fallback for U,
so it costs us a data point.

Target U rate: **<5%**.

---

## What you should NOT see / use

To keep the annotation independent, the file you have hides:
- Our automated rule-based classification
- The LLM-judge's classification
- The sampling stratum
- The research hypothesis
- **The other annotator's labels**

If you happen to know any of these from a previous task, **please put
them out of mind while labeling**. Label only on the operational
criteria above. The point of the annotation is precisely to provide
two labeling sources that do not depend on any of the above — or on
each other.

---

## When you're done

1. Send the file back **as-is, filename unchanged**
   (`data_to_label_annotator1.csv` or `data_to_label_annotator2.csv`)
   — the number in the filename identifies which annotator it came
   from.
2. Quick self-check: how many rows are labeled U? If >10%, re-read
   the U-criterion above and reconsider those rows.
3. Hand back the file. **Only after both annotators have handed in**,
   the analysis script (`analyze_when_done_n500.py`) computes
   inter-annotator agreement (Cohen's κ) and exports the rows where
   you two disagreed.
4. Joint adjudication session: go through the disagreement rows
   together, agree on a `consensus_label` for each, and return the
   filled `disagreements_for_adjudication.csv`.

---

## Quick-reference card (print this)

```
Q1.  Final answer correct?
       GSM8K, BBH  : extracted_answer  vs  reference_answer
       MMLU        : extracted_answer  vs  correct_letter
       (ignore reference_answer for MMLU — it's the option text)
       (extracted_answer empty? read it off the continuation)
       NO  → C
       YES → Q2

Q2.  Wrong step, correct final answer. Did it:
       A) ignore the wrong step, continue as if it was correct
       B) notice the error and correct it
       B if: correction language targeting the injection, OR
             it engages / rejects the injected claim
             (option sweep calling the injected option wrong counts)
       A if: math shortcut with no mention of the injected value;
             independent path that never touches the claim;
             injected step named the correct option and it agrees
       When in doubt → A

Unclear → U  (use sparingly, <5%)

Confidence:
  high   = one obvious answer
  medium = one likely answer, others plausible
  low    = need a second opinion

Independence: no discussing rows with the other annotator
until both files are handed in.
```
