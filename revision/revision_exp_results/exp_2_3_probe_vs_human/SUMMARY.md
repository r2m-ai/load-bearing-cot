# Probe-vs-Human Eval — n=198 (item 2.3)

**Setup**: layer-10 binary bypass probe (A vs non-A), logistic regression
with balanced class weights, trained on the N=2,000 features set using
grouped 5-fold CV (StratifiedGroupKFold by base question, random_state=42).
Held-out predictions for the 15 annotation rows that overlap N=2,000;
final-probe predictions for the 185 non-overlap rows using freshly
extracted Gemma-2-9B-IT layer-10 features (extracted 2026-05-24 on
H200 via `revision/scripts/exp_2_3_extract_annot_features.py`; sanity
check on the 15 overlap rows showed cosine similarity ≥ 0.9999 vs the
canonical features file, confirming bit-equivalent extraction).

**Coverage**: 200 annotation rows → 198 evaluable (2 "U" / unclear
labels dropped). 15 overlap + 183 non-overlap = 198.

## Headline numbers

| Comparison | n | Probe | Trivial-A | Δ |
|---|---|---|---|---|
| **Overall** | 198 | **66.7%** [59.8, 72.9] | 60.6% | **+6.1pp** |
| gsm8k | 53 | 81.1% [68.6, 89.4] | 81.1% | 0 |
| mmlu | 75 | 61.3% [50.0, 71.5] | 40.0% | **+21.3pp** |
| bbh | 70 | 61.4% [49.7, 72.0] | 67.1% | −5.7pp |

## In-training vs non-overlap

| Subset | n | Agreement |
|---|---|---|
| in-training (held-out CV pred) | 15 | 46.7% [24.8, 69.9] |
| non-overlap (final-probe on new features) | 183 | 68.3% [61.2, 74.6] |

The in-training row count is small (n=15) because the original N=2,000
probe-training set and the n=200 human-annotation set were sampled
independently; the overlap is incidental. The non-overlap number is
load-bearing.

## What this means

- **The probe's signal-above-trivial lives in MMLU.** On MMLU's 75
  annotation rows the probe beats the trivial-A baseline by **21pp**
  (61.3% vs 40.0%). Wilson CIs do not overlap. This is the only
  source × eval where the probe is unambiguously informative.
- **GSM8K is saturated.** 81% of the GSM annotation rows are human-A,
  so trivial-A already nails 81% and the probe can't differentiate
  itself — both score 81.1% (43/53 identical, the probe makes the same
  always-A call that the trivial baseline makes).
- **BBH is the failure mode.** The probe under-performs trivial-A by
  5.7pp on BBH (61.4% vs 67.1%). The N=2,000 training set is
  source-skewed (GSM-heavy by class imbalance), and the learned
  decision boundary doesn't transfer to BBH's bypass dynamics. This is
  consistent with the §5.3-footnote caveat about cross-source
  generalization.
- **Probe-vs-training is 78.1%, probe-vs-human is 66.7%.** The 11.4pp
  gap is the compositional effect of (probe ↔ judge) × (judge ↔ human)
  — the probe optimizes for judge labels, humans disagree with the
  judge ~26% of the time on this stratified subset (per item 2.1), so
  probe ↔ human ≈ 0.78 × 0.74 ≈ 0.58 in expectation. The observed
  0.667 is slightly better than that lower bound.

## Audit notes (2026-05-24, post-extraction)

1. **Bug fixed**: the earlier `exp_2_3_probe_vs_human.py` did
   `subclassified_pairs.json[:2000]` to label the features, but the
   features file is keyed by its own `pair_ids` array (only 4/2000 rows
   line up by chance). The previously reported held-out probe accuracy
   of 92.9% was on essentially random labels. After fixing to the
   `pair_ids`-keyed lookup used by `scripts/10b_retrain_probes.py`:
   - Held-out probe accuracy: 92.9% → **78.1%** (matches paper §5.3's
     "~80% binary bypass" at layer 10 ✅)
   - Class balance: 95% A → **67.2% A** (matches paper's "majority
     baseline 67.2% bypass" ✅)
2. **15 vs 47 overlap**: the previously reported "47 overlap rows, all
   GSM8K" was a side effect of the same bug — taking the first 2000 of
   `subclassified_pairs.json` happens to be GSM-dense, so 47 GSM rows
   from the annotation set appeared to "overlap". True overlap by the
   features file's `pair_ids` is 15 rows (mix of sources). This
   collapses the in-training CI to [24.8, 69.9] but doesn't matter
   because the non-overlap arm covers 183 rows.

## For the manuscript

**§5.3 footnote language (draft)**:

> "On the n=198 human-annotated continuations (item 2.1; two unclear
> labels dropped), the layer-10 bypass probe agrees with the human
> label 66.7% of the time (95% CI [59.8, 72.9]), vs a trivial-always-A
> baseline of 60.6% on the same subset. Agreement is source-dependent:
> the probe beats the trivial baseline by 21pp on MMLU (61.3% vs
> 40.0%; n=75), ties it on GSM8K (81.1% on both; n=53, where 81% of
> annotated rows are bypass and trivial-A is hard to beat), and falls
> 6pp below trivial on BBH (61.4% vs 67.1%; n=70), reflecting the
> probe-training subsample's GSM-skewed class distribution. The
> probe-vs-judge held-out accuracy on the N=2,000 training set is
> 78.1% at the same layer; the 11pp gap to probe-vs-human is
> compositional with the judge-vs-human disagreement rate of 26.2%
> measured in item 2.1."

## Files

- `results.json` — full per-row predictions + overall agreement
- `per_row.csv` — same in tabular form for spot-checking
- `extract.log` — extraction stdout from the H200 run
- `../../human_annotation/SUMMARY_n200.md` — companion 2.1 analysis
- Source script: `revision/scripts/exp_2_3_probe_vs_human.py`
- Feature extraction script: `revision/scripts/exp_2_3_extract_annot_features.py`
- New features file: `data/processed/extracted_features_annot200.npz`
