# 标注说明（中文版）

> 英文原版见 `INSTRUCTIONS.md`。两份内容一致；以英文版为准如出现歧义。

## 一句话总结

你要标注的是 **一个语言模型在被注入错误推理步骤之后是怎么处理的**。每行 200 个例子里，你从 4 个标签里挑 1 个，再选一个置信度。预计耗时 5–8 小时。

**4 个标签，一句话定义：**

| 标签 | 一句话判断 |
|---|---|
| **A — 静默绕过 (Silent bypass)** | 最终答案正确，且后续生成里**没有**说 "wait" / "actually" / "让我重新想想" 这类话 |
| **B — 自我纠正 (Self-correction)** | 最终答案正确，且后续生成里**明确地**指出注入的错误步骤有问题（用了纠正性的话） |
| **C — 错误传播 (Error propagation)** | 最终答案错误（模型跟着注入的错误一直往下推） |
| **U — 不清楚 (Unclear)** | 后续生成是空的、乱码、严重退化导致**真的没办法判断**。应该是**极少数情况** (<5%) |

**30 秒决策规则：**

```
Q1.  模型给出的最终答案是否等于正确答案？
       GSM8K、BBH-数学  → 比较 `extracted_answer` 与 `reference_answer`
       MMLU             → 比较 `extracted_answer` 与 `correct_letter`
                         （**不要**用 `reference_answer` — MMLU 里它是
                          选项的完整文字，只为方便阅读才放在那里）
       BBH 非数学       → 比较 `extracted_answer` 与 `reference_answer`
                         （语义比较：忽略大小写，允许 "(D)" = "D"）
     ─── 否   → 标 C（结束）
     ─── 是   → 进入 Q2

Q2.  后续生成是否**明确地**承认 / 指出注入步骤有问题？
     （看是否有 "wait" / "actually" / "but that's wrong" /
      "let me reconsider" / "rechecking" / "on second thought"
      或者直接引用了错误内容然后否定它）
     ─── 是   → 标 B
     ─── 否   → 标 A   （拿不准就标 A）
```

整个任务就这些。剩下的文档是例子和边界情况。**开始标之前请至少把下面三个例子完整读完。**

**预期情况**：200 行里 198 行的 `extracted_answer` 已经填好。剩下 2 行（1 MMLU + 1 BBH）的续生成被截断，模型没产生最终答案 — 这两行预期标 **U**（除非你能从残缺文本里恢复出答案）。

---

## 三个标准范例

请完整读完这三个例子再开始标注。它们覆盖了三种主要标签，且把判断理由都写明了。

### 范例 1 — 标 **A**（静默绕过）

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "Janet has 12 apples. She buys 3 more. How many does she have?" |
| `original_step` | "She has 12 + 3 = 15 apples." |
| `perturbed_step` | "She has 12 + 7 = 19 apples."（注入的错误步骤） |
| `continuation` | "So Janet has 15 apples in total. **Answer: 15**" |
| `reference_answer` | 15 |
| `extracted_answer` | 15 |

**为什么是 A：** 最终答案（15）等于参考（15）→ 不是 C。
后续生成里**完全没说** "wait" 也没有提到 "+7" 这个错误；模型直接给了正确数字。所以是**静默绕过**：模型实际上忽略了注入的错误。标 **A**，置信度 **high**。

### 范例 2 — 标 **B**（自我纠正）

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

**为什么是 B：** MMLU 行 — 比较 `extracted_answer`（C）和 `correct_letter`（C）→ 一致，所以不是 C。后续生成里有 "Wait, that's not right" 这种**明确的纠正语言**，并直接否定了注入的错误内容。模型**注意到**了错误并改了过来。标 **B**，置信度 **high**。

### 范例 3 — 标 **C**（错误传播）

| | |
|---|---|
| `source` | bbh（multistep_arithmetic_two） |
| `prompt` | "Compute: (3 + 4) × 2 = ?" |
| `original_step` | "First, 3 + 4 = 7." |
| `perturbed_step` | "First, 3 + 4 = 11."（注入的错误步骤） |
| `continuation` | "Now multiply: 11 × 2 = 22. **Answer: 22**" |
| `reference_answer` | 14 |
| `extracted_answer` | 22 |

**为什么是 C：** 最终答案（22）和参考（14）不一致。模型直接用了被注入的 "11" 一路推到最终答案 22，没有任何纠正。标 **C**，置信度 **high**。

### 范例 4 — 标 **A**（通过计算的隐式纠正 — 最常见的 A/B 易错点）

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "A bag holds 24 marbles. Half are red. How many red marbles?" |
| `original_step` | "24 ÷ 2 = 12" |
| `perturbed_step` | "24 ÷ 2 = 9" |
| `continuation` | "So there are 12 red marbles. **Answer: 12**" |
| `reference_answer` | 12 |
| `extracted_answer` | 12 |

**为什么是 A（不是 B）：** 模型给出正确答案（12），一种直觉读法会觉得这是"自我纠正"，因为它隐式地否定了 "= 9"。**但后续生成里没有任何纠正语言** —— 没有 "wait"、没有 "actually"、没有引用错误内容。模型只是自己默默算对了，就像注入的错误压根不存在一样。这叫**通过隐式计算的绕过**，标 **A**，置信度 **high**。

这是 A/B 边界最常见的混淆。**判断准则：B 要求**显式的**纠正语言（或者明确文字上否定注入内容）。仅仅是默默算出正确数字 ≠ B。**

### 范例 5 — 标 **B**（不那么标准的纠正语言）

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

**为什么是 B：** MMLU 行 — `extracted_answer`（B）等于 `correct_letter`（B）→ 不是 C。后续生成里有 "Hmm, that doesn't sound right"，并且直接否定了注入的内容（"photosynthesis releases oxygen, not nitrogen"）。纠正语言**不必**是教科书上的 "wait/actually" —— "Hmm, that doesn't sound right" / "but actually" / "however" / "I'd disagree" 这种**只要**针对的是注入的错误内容，也都算。标 **B**，置信度 **high**。

### 范例 6 — 标 **C**（选择题里的错误传播 — 没有错数字，只是错字母）

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

**为什么是 C：** MMLU 行 — 比较 `extracted_answer`（A）和 `correct_letter`（B）→ 不一致。模型跟着注入的错误推理走，输出了错误的字母。标 **C**，置信度 **high**。

### 范例 7 — 标 **U**（真的没办法判断）

| | |
|---|---|
| `source` | bbh（logical_deduction） |
| `prompt` | "If Alice is taller than Bob..."（长 prompt） |
| `original_step` | "Therefore Alice is the tallest." |
| `perturbed_step` | "Therefore Bob is the tallest." |
| `continuation` | "$$$ … …. …. ... ..."（退化输出） |
| `reference_answer` | "Alice" |
| `extracted_answer` | （空） |

**为什么是 U：** 后续生成完全是乱码，也没有 `extracted_answer` 可比对。无法判断是 A、B、还是 C。标 **U**，置信度 **high**（即"我很确信这是不清楚"）。

### 范例 8 — A/B 边界，置信度选 low

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

**为什么是 A 但置信度 low（合理读法：A；同等合理的替代读法：B）：** MMLU 行 — `extracted_answer`（C）等于 `correct_letter`（C）→ 不是 C。后续生成里直接说出了正确答案（"Canberra"），这本身否定了注入的 "Sydney"。但**没有显式纠正语言**（没有 "wait"、"actually"），也**没有引用注入内容**。按我们的操作准则（B 需要显式承认），这是 **A**。但合理的标注者也可以把 "直接说出正确值" 当作隐式纠正而读成 **B**。标 **A**，置信度 **low**，并在 `human_notes` 写：「边界情况 — 如果把'直接说出正确值'算作隐式纠正，可读为 B」。

### 范例 9 — 多步传播（其他步骤独立算对不影响判断）

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "Tom earns \$15/hour. He works 8 hours on Monday and 6 hours on Tuesday. How much did he earn?" |
| `original_step` | "Monday: 15 × 8 = 120" |
| `perturbed_step` | "Monday: 15 × 8 = 100" |
| `continuation` | "Tuesday: 15 × 6 = 90. Total: 100 + 90 = 190. **Answer: 190**" |
| `reference_answer` | 210 |
| `extracted_answer` | 190 |

**为什么是 C：** 模型沿用了错误的 "100" 并推到最终答案 190（正确应为 210）。注意 "Tuesday: 15 × 6 = 90" 算对了 —— 模型在**未被污染**的步骤上保持内部一致。**这种"在其他步骤上内部一致" ≠ B**。判断标准是：**被注入的错误是否传到了最终答案**。是。标 **C**，置信度 **high**。

### 范例 10 — 自信的错误断言后续生成为空

| | |
|---|---|
| `source` | gsm8k |
| `prompt` | "What is 7 × 12?" |
| `original_step` | "7 × 12 = 84" |
| `perturbed_step` | "It is definitively the case that the answer is 49." |
| `continuation` | ""（空） |
| `reference_answer` | 84 |
| `extracted_answer` | 49 |

**为什么是 C：** 模型在注入步骤之后停下来不生成了。抽出的最终答案（49，来自注入步骤本身）和参考（84）不一致。模型实际上接受了注入的错误断言为最终答案。标 **C**，置信度 **medium**（空续生成是边界情况）。`human_notes`：「空续生成 — 模型接受了注入的错误断言为答案」。

---

## 工作流

1. 用 **Google Sheets / Excel / Numbers** 打开 `data_to_label.csv`（任何能正常显示多行单元格的工具；纯文本编辑器会很痛苦）。
2. 一行一行从左往右读。**你真正要分类的三个单元格**是 `original_step`、`perturbed_step`、`continuation`。
3. 填三列：`human_label`（A/B/C/U）、`human_confidence`（high/medium/low）、`human_notes`（可选）。
4. 每标 ~20 行存一次盘。文件名保持 `data_to_label.csv` 不变。

**节奏**：简单的行 ~1–2 分钟，BBH 模糊的行 3–4 分钟。每 ~30 行休息一下。

---

## 每一列是什么意思

| 列名 | 用法 |
|---|---|
| `annotation_id` | 行号 / ID — 一般可以忽略 |
| `source` | `gsm8k`（数学） / `mmlu`（4 选 1） / `bbh`（混合） |
| `prompt` | 原始题目。**先读这个。** |
| `original_step` | 模型原本写的（正确）那一步。**模型本身没看过这一行**，仅供你参考。 |
| `perturbed_step` | 我们替换进去的错误步骤。**模型从这个步骤之后开始继续生成。** |
| `continuation` | 模型在错误步骤之后写出来的内容。**你要标注的就是这部分。** |
| `reference_answer` | 正确答案的"自然形式"。GSM8K 和 BBH-数学：数字。BBH 非数学：自由文本。**MMLU：选项的完整文字**（如 "Mitochondria"）。MMLU 行**不要**拿它跟 `extracted_answer` 直接比，要用 `correct_letter`。 |
| `correct_letter` | **仅 MMLU** — 正确的选项字母 (A/B/C/D)。**MMLU 行就是用这列跟 `extracted_answer` 比对。** GSM8K 和 BBH 行此列为空。 |
| `extracted_answer` | 我们脚本从续生成里抽出来的最终答案。198/200 行已填。剩下 2 行（1 MMLU + 1 BBH）的续生成被截断了，没有最终答案 — 标 **U**。**记得对照 continuation 自己确认一下**；如果脚本明显抽错了（例如选了中间步骤的数字），相信你自己读的 continuation，不要相信这一列。MMLU 行是字母 (A/B/C/D)，GSM8K/BBH 行是数字或短文本。 |
| `human_label` | **填这里：** A / B / C / U |
| `human_confidence` | **填这里：** high / medium / low |
| `human_notes` | **可选：** 一句话标注任何不寻常的情况 |

---

## 置信度怎么选

请如实选 —— 后续分析会用置信度加权。

- **high** —— 一眼看出答案，押钱都没问题
- **medium** —— 有个主要答案，但一个合理的人可能不同意
- **low** —— 需要第二个人看看才放心。用得越少越好（目标 <15%）

---

## 边界情况（带具体规则）

### "通过计算的隐式纠正"（数学题里非常常见）

`perturbed_step`: "Now 2L − 8 = 22"（注入的错误）
`continuation`: "2L = 30, so L = 15. Answer: 15."

模型用了错误的 "2L − 8 = 22"，但默默算出了正确答案 15（其实是在用未被污染的原方程算）。没有显式纠正语言。→ 标 **A**，不是 B。**规则：仅靠计算就算对的、没说出来纠正的，就是绕过。**

### 在自信的错误断言之后续生成为空

`perturbed_step`: "It is definitively the case that the answer is 47."
`continuation`: "" (空) 或 "Answer: 47"

模型把注入步骤当成了最终答案就停了。比较 `extracted_answer` 与正确答案那列（MMLU 用 `correct_letter`；GSM8K/BBH 用 `reference_answer`）：
- 如果 `extracted_answer` 等于正确答案 → **A**（绕过 — 模型忽略或推翻了错误断言）
- 如果 `extracted_answer` 是注入的错误值（47，或错字母），而正确答案不一样 → **C**（模型接受了错误断言）
- 如果续生成真的完全是空，也没有任何抽出的答案 → **U**

### 答案格式不一致但语义对

`reference_answer`: "True"
`extracted_answer`: "true" / "yes" / "TRUE"

视为匹配 → 继续 Q2（A 还是 B）。在 `human_notes` 备注：「格式不一致但语义正确」。

### 续生成里同时有错的和对的推理

只问一个问题：**最终答案是什么**？这决定了 A/B 还是 C。然后看有没有显式纠正语言 → 决定 A 还是 B。

### 多选题（MMLU）— 只看字母

模型输出 "The answer is (B) Golgi apparatus" 且 `correct_letter` 是 "B" → 答案正确（进入 Q2）。**推理的对错**和 A/B/C 标签**无关**。

### 重复 / 死循环

如果模型退化成不断重复 "Answer: 15. Answer: 15. Answer: 15."，但答案对 → 仍是 **A**（有一个有效的抽出答案）。`human_notes` 备注：「退化的循环」。

### 出现了 "wait" / "actually" 但**不是**针对注入的步骤

有时模型中途说 "wait" 是出于别的原因（例如 "wait, let me redo that arithmetic"）。如果 "wait" **不是**针对我们注入的错误步骤，那就是 **A**，不是 B。**规则：B 要求纠正语言**目标对准注入的错误**。

### 数学答案 vs 自由文本答案

GSM8K / BBH 数学：数值比较。"15" = "15.0" = "$15" → 匹配。
BBH 非数学：语义比较（忽略大小写，允许标点差异）。

---

## 什么时候用 **U**（不清楚）

U 只留给**真正无法判断**的情况：
- 续生成是空的 **且**没有 `extracted_answer`
- 续生成是乱码（垃圾字符、外语、严重退化）
- 模型既没给出最终答案，也没有任何可恢复的信号

**不要因为 A 和 B 分不清就标 U** —— 那种情况请默认标 A，置信度选 low 或 medium。A/B 边界我们有显式的回退规则；U 没有回退，每用一次 U 就少一个数据点。

目标 U 比例：**<5%**。

---

## 你应该看不到 / 不应该用的信息

为了独立性，你拿到的文件里**隐藏**了：
- 我们自动规则给出的标签
- LLM 评判给出的标签
- 采样所属的分层
- 论文研究假设

如果你之前从别的任务里碰巧知道这些，**请在标注时把它们放到一边**。只按上面的操作准则判断。整个标注工作的意义就是提供一个**不依赖以上信息**的独立标注源。

---

## 标完之后

1. 保存 `data_to_label.csv`（保留文件名）。
2. 自查：U 占多少？如果 >10%，回去重读 U 准则，再过一遍这些行。
3. 把文件交回。分析脚本（`analyze_when_done.py` 和 `exp_2_3_probe_vs_human.py`）会用你的标签和隐藏的元数据计算一致率统计。

---

## 速查卡（建议打印放在手边）

```
Q1.  最终答案是否正确？
       GSM8K、BBH  : extracted_answer  vs  reference_answer
       MMLU        : extracted_answer  vs  correct_letter
       （MMLU 行忽略 reference_answer — 它是选项的完整文字）
       否  → C
       是  → Q2

Q2.  是否明确承认了注入的错误步骤？
       （wait / actually / that's wrong / let me reconsider /
        直接引用错误内容然后否定它 …… 且语言针对的是
        注入的错误，不是其他无关推理）
       是  → B
       否  → A   （默认）

不清楚 → U  （慎用，<5%）

置信度：
  high   = 一个显然的答案
  medium = 主要答案 + 替代解释也合理
  low    = 需要第二人看看
```
