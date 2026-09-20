# From Decorative to Load-Bearing: Task Difficulty Shapes the Causal Role of Chain-of-Thought

**Renee Jia, Di Mu** · [R2M AI](https://r2m.ai/) · Transactions on Machine Learning Research, 2026

[📄 Paper](paper/main.pdf) · [🤗 Dataset](https://huggingface.co/datasets/ReneeJia/cot-load-bearingness) · [📊 Findings](#-what-we-found) · [⚡ Quickstart](#-run-the-benchmark) · [🔬 Reproduce](docs/paper-reproduction.md) · [🗺️ Experiment map](experiments/README.md)

> **When a language model writes a chain of thought, does that reasoning actually constrain its answer?**

It matters because chain-of-thought is widely proposed as a way to monitor what models are doing — and a monitor that reads the chain is only informative if the chain drives the output.

We test it directly: **change one step mid-chain, cut the trace there, and force the model to continue.** What it writes next shows whether it ignores the injected error, catches and repairs it, or carries it straight into the answer.

**The answer depends on difficulty.** On easy tasks the written chain is close to decorative. On hard ones it is load-bearing.

<p align="center">
  <img src="docs/assets/research_overview.svg" width="1080" alt="Study overview: a correct multi-step baseline, a single-step perturbation and truncation, then a forced continuation, which falls into silent bypass (45.3%), self-correction (26.9%) or error propagation (27.8%). The paper then tests difficulty and perturbation controls, validates the labels, and examines hidden-state probes and steering.">
</p>

<sub>Pooled Gemma results under the production judge (n = 21,238). **A** and **B** both land on the correct answer and differ only in whether the model visibly notices the error — a distinction sensitive to the judge prompt. **C** does not, and is prompt-invariant. The protocol tests the **CoT → answer** link, not whether the trace mirrors the model's internal computation.</sub>

---

## 📊 What we found

<p align="center">
  <img src="docs/assets/research_findings.svg" width="1000" alt="Panel a: error propagation rises as baseline accuracy falls, for Gemma, Llama and DeepSeek. Panel b: across 29 MMLU subjects error propagation spans 7.5 to 53.7 percent at fixed question format. Panel c: holding perturbation type fixed, error propagation rises from 3.9 percent on GSM8K to 64.5 percent on BBH multistep arithmetic; 98.8 percent of explained deviance goes to task difficulty.">
</p>

### 1 · Harder for the model → more load-bearing

Error propagation climbs **3.9% → 22.3% → 40.9%** across GSM8K, MMLU and BBH as Gemma's own accuracy falls from 86.5% to 57.9%. The cleanest cut is *inside* MMLU, where the question format and every perturbation strategy are held fixed and the rate still spans **7.5% to 53.7%** across 29 subjects.

| Model | GSM8K | MMLU | BBH |
|---|:---:|:---:|:---:|
| **Gemma-2-9B-IT** | 3.9% <sub>*86.5% acc.*</sub> | 22.3% <sub>*74.8%*</sub> | **40.9%** <sub>*57.9%*</sub> |
| **Llama-3.1-8B-Instruct** | 12.4% <sub>*55.3%*</sub> | 62.8% <sub>*40.8%*</sub> | **65.5%** <sub>*40.1%*</sub> |
| **DeepSeek-R1-Distill-Qwen-7B** | 5.1% <sub>*38.5%*</sub> | 2.7% <sub>*52.4%*</sub> | 16.8% <sub>*20.2%*</sub> |

<sub>Error propagation, with baseline accuracy underneath. Difficulty here is relative to *the model's own* competence, not an absolute property of the task.</sub>

### 2 · It's the difficulty, not the kind of edit

Hold the perturbation type fixed at numerical and propagation still rises **16×** from GSM8K to BBH multistep arithmetic (3.9% → 64.5%). Text edits on GSM8K land at 6.5%. A regression over all **28,584** continuations assigns **98.8%** of explained deviance to task difficulty against **0.8%** to perturbation type. → [analysis](results/controls/variance_decomposition/results.json)

### 3 · Reasoning-trained models are a different story

Llama-3.1-8B-Instruct reproduces the gradient at a lower accuracy level. DeepSeek-R1-Distill propagates far fewer errors and shows a **compressed** gradient — consistent with reasoning-specific post-training playing a role, though our setup can't isolate its causal effect.

### 4 · We can read the behavior, but not steer it

Probes hit **75.3%** three-class and **86.2%** error-propagation accuracy under question-grouped 5-fold CV. Steering along those same directions is a **null result** across 11,556 generations. An eight-direction intervention moves just **24.6%** of Type C cases at its strongest setting (14/57; 95% CI 15.1–37.1%).

### 5 · Label quality bounds all of the above

Two blind annotators agree at **κ = 1.00** on the C-vs-non-C split and κ ≈ 0.93 on the full scheme. Against the paper's production labels: **96.2%** agreement on the binary call, but only **76.0%** three-way — the A/B boundary is genuinely prompt-sensitive. → [human-study materials](data/annotations/)

<details>
<summary><b>📐 How to read these numbers</b></summary>

<br>

- **The binary split is the load-bearing one.** Every claim above rests on C-vs-non-C, which is invariant across all four judge prompts by construction and on which both annotators agreed for every row. Treat any A/B-dependent figure as conditional on the judge prompt.
- **Within-MMLU is the primary evidence.** The cross-dataset gradient also reflects perturbation design and residual BBH answer-scoring errors, so we lean on the fixed-format contrast instead.
- **Deviance shares are model-specific.** The 98.8% / 0.8% split is specific to the fitted model and partition. Perturbation type can still matter a great deal *within* a single task — a magnitude sweep rules out a competence-floor reading.
- **DeepSeek isn't a clean replication.** It filters to ≥4 post-think steps (n = 1,603) and edits only the visible rationale after `</think>`, never the thinking trace. Cells aren't comparable to Gemma's n = 21,238; the cross-model ordering is.
- **Steering conclusions are narrow.** They cover the additive interventions we tested, not steerability in general. Type B remains underpowered throughout.
- **Labels cap probe accuracy.** Refitting the same probes on corrected labels drops three-class accuracy from 99.9% to 75.0% — a caution for any behaviorally labeled probe.

</details>

The [paper](paper/main.pdf) has the full methods, uncertainty estimates and limitations. The [experiment map](experiments/README.md) links every analysis to its code and released artifacts.

---

## ⚡ Run the benchmark

**Measure whether a model's chain-of-thought is causally load-bearing.** Python 3.10+, in an environment supported by vLLM:

```bash
git clone https://github.com/r2m-ai/load-bearing-cot.git
cd load-bearing-cot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python evaluate.py --model deepseek-r1-distill-qwen-7b --dataset gsm8k \
  --output-dir runs/deepseek-gsm8k
```

The default run samples **100 questions** with a fixed seed, uses greedy decoding, and tests early, middle and late interventions with a 4,096-token budget per generation. Use `--all` for a full dataset run. No external judge API is called.

```bash
# other datasets, tasks and strategies
python evaluate.py --model gemma-2-9b-it --dataset mmlu

python evaluate.py --model llama-3.1-8b-instruct --dataset bbh \
  --task multistep_arithmetic_two --strategy arithmetic

# any vLLM-compatible model, by Hugging Face ID or local path
python evaluate.py --model Qwen/Qwen2.5-7B-Instruct --dataset gsm8k
```

A model needs vLLM inference, a chat template, and continuation from an assistant prefix. Accept the model license and authenticate with Hugging Face before loading gated weights. The paper ran on a single **NVIDIA H100 80 GB**. See `python evaluate.py --help` for revision pins, local JSONL input and parallelism.

> ⚠️ **Implementation status.** `load-bearing-v0.1` has offline tests for scoring, continuation setup and output handling, but **real GPU runs of this entry point are not yet validated**. The published Gemma, Llama and DeepSeek results come from the paper pipeline. Qwen is an example of how to plug in another model, not a reported result.

---

## 📖 Interpreting a run

The evaluator keeps correctly answered baselines with at least three visible reasoning lines, edits one line, truncates there, and generates both a perturbed continuation and an unedited-prefix control.

**Error propagation rate** is the fraction of parseable perturbed continuations with an incorrect final answer — conditional on baseline correctness, eligibility and the chosen intervention. Low propagation may mean either bypass *or* correction, and high propagation is **not** a measure of reasoning quality. The evaluator reports the correctness-based C-vs-non-C proxy only; confirming that an error was genuinely taken up means reading the trace.

| Output file | Contents |
|---|---|
| `summary.json` | Propagation rate, paired control error and rate difference, baseline accuracy, sample counts, answer coverage; broken down by task and position |
| `baselines.jsonl` | Questions, references, prompts, baseline responses, eligibility |
| `interventions.jsonl` | Original and edited steps, both prefixes, continuations, extracted answers and scores |
| `manifest.json` | Protocol version, arguments, model ID, dataset fingerprints, sampled IDs/hash, package versions, chat template, run status |

Unparseable answers stay unscored and undefined rates are `null`. Output directories are never overwritten. When comparing models, report coverage and eligibility alongside the rate and keep the intervention and parsing settings fixed. The [protocol spec](docs/protocol.md) defines the estimand and how it differs from the paper experiments.

---

## 📁 Repository

| Path | Contents |
|---|---|
| [`evaluate.py`](evaluate.py) · [`load_bearing/`](load_bearing/) | Reusable evaluator: model backend, dataset adapters, intervention and scoring |
| [`experiments/`](experiments/) | The paper pipeline and every analysis built on it, grouped by research question · [map](experiments/README.md) |
| [`results/`](results/) | Released result snapshots, grouped the same way |
| [`data/`](data/) | Example records, reference artifacts and human-annotation materials |
| [`paper/`](paper/) · [`figures/`](figures/) | Manuscript source and PDF, paper figures and table CSVs · [map](figures/README.md) |
| [`docs/`](docs/) | [Protocol spec](docs/protocol.md) and [reproduction guide](docs/paper-reproduction.md) |
| [`tests/`](tests/) | Offline regression tests — `python -m unittest discover -s tests` |

🤗 The paper's full record set is published as a Hugging Face dataset, [**`ReneeJia/cot-load-bearingness`**](https://huggingface.co/datasets/ReneeJia/cot-load-bearingness): every labeled continuation from all three models, the matched perturbation controls, every steered generation, four-variant judge labels and the paper's aggregate tables. The headline numbers are recomputable from those rows.

Git keeps the generated records out of this repository, so clone the dataset rather than this repository if you want the rows themselves. Hidden-state arrays and probe checkpoints are in neither — the [reproduction guide](docs/paper-reproduction.md) lists what ships and what has to be regenerated. See [Contributing](CONTRIBUTING.md) for adding models, datasets or results.

---

## 📝 Citation

```bibtex
@article{jia2026loadbearing,
  title   = {From Decorative to Load-Bearing: Task Difficulty Shapes the Causal Role of Chain-of-Thought},
  author  = {Renee Jia and Di Mu},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```

Please report the protocol version and evaluation settings alongside any number you publish.

Code: [MIT](LICENSE) · Paper text and figures © the authors · Datasets and model weights keep their own terms.

Questions and contributions welcome — Renee Jia, reneejia@r2m.ai · [r2m.ai](https://r2m.ai/)
