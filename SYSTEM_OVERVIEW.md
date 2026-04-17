# WSI-Agents: Multi-Agent Medical WSI Analysis System

## Overview

This system implements the **WSI-Agents** framework described in:

> *WSI-Agents: A Collaborative Multi-Agent System for Multi-Modal Whole Slide Image Analysis*

**Core idea:** Given a pathology question (with its WSI already analyzed by several MLLMs offline), the pipeline takes those pre-stored answers, runs three parallel verification agents, then an integration agent scores, selects, and iteratively refines the best response.

> The system does **not** call vision models at runtime. MLLM answers are pre-computed and stored in JSONL files. At inference time the pipeline only calls the LLM (GPT-4o) for reasoning and verification steps.

---

## Quick Start

```bash
# Set API credentials
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Run a single task
python MedicalAnalysisSystem.py Stage Stage-wsi-vqa-answer.jsonl

# Run one demo case per task category (outputs to demo_output/)
python run_demo.py

# Run all tasks (auto-skips already-processed cases)
python run_experiments.py

# Run specific task types
python run_experiments.py Morphology Treatment

# Preview remaining work without running
python run_experiments.py --dry-run
```

---

## Installation

**Python 3.10+ required.**

```bash
# 1. Clone the repo
git clone <repo-url>
cd open_source

# 2. Create and activate conda environment
conda create -n WSI_Agents python=3.10 -y
conda activate WSI_Agents

# 3. Install dependencies
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
|---------|---------|
| `autogen-core` | Multi-agent runtime (message routing, agent registration) |
| `autogen-ext[openai]` | OpenAI chat completion client |
| `langchain` + `langchain-openai` | LLM chains for knowledge base QA |
| `langchain-chroma` + `chromadb` | Chroma vector store for medical KB |
| `python-docx` | Parse WHO Classification `.docx` books into KB chunks |
| `pydantic` | Message / result model definitions |

**Set API credentials before running:**

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # optional, defaults to OpenAI
```

**Build the knowledge base (one-time):**

```bash
# Place WHO Classification .docx files in knowledge_base_demo/who_books/
python knowledge_base_demo/build_kb.py
# Outputs to: ../medical_kb_structured/  (Chroma vector DB)
```

---

## Configuring Models (Key-Value Mapping)

The set of MLLMs is fully configurable. In `config.py`, `configure_task()` builds a `MLLM_PATHS` dict that maps a **model key** to its **answer file**:

```python
# config.py — MLLM_PATHS are derived automatically from the task name
# For task_name="Treatment", configure_task builds:
MLLM_PATHS = {
    "mllm_1": ".../MLLMs_output/Treatment-mllm1-answers.jsonl",
    "mllm_2": ".../MLLMs_output/Treatment-mllm2-answers.jsonl",
    "mllm_3": ".../MLLMs_output/Treatment-mllm3-answers.jsonl",
    "mllm_4": ".../MLLMs_output/Treatment-mllm4-answers.jsonl",  # fallback to mllm3 if absent
}
```

The keys (`mllm_1`, `mllm_2`, …) are just labels used internally. Each Expert Agent reads a specific subset of these keys; to override a path at runtime use `Config.update_mllm_path("mllm_1", new_path)`.

### Task-specific model assignments

All four task types use the same pipeline with three MLLM inputs:

| Task route | Keys read | Output path |
|---|---|---|
| **Morphology** | `mllm_1`, `mllm_2`, `mllm_3` | → Full verification pipeline |
| **Diagnosis** | `mllm_1`, `mllm_2`, `mllm_3` | → Full verification pipeline |
| **Treatment** | `mllm_1`, `mllm_2`, `mllm_3` | → Full verification pipeline |
| **Report** | `mllm_1`, `mllm_2`, `mllm_3` | → Full verification pipeline |

To run a specific task:

```python
Config.configure_task("Treatment", "Treatment-questions.jsonl")
# Config.QUESTIONS_PATH       == ".../questions/Treatment-questions.jsonl"
# Config.MLLM_PATHS["mllm_1"] == ".../MLLMs_output/Treatment-mllm1-answers.jsonl"
# Config.MLLM_PATHS["mllm_3"] == ".../MLLMs_output/Treatment-mllm3-answers.jsonl"
```

---

## System Inputs

For each question the pipeline receives:

| # | Input | Source | Used by |
|---|-------|--------|---------|
| 1 | Pathology question text | `questions/<Task>-questions.jsonl` → `prompt` field | all agents |
| 2 | MLLM_1 pre-stored answer | `MLLMs_output/<Task>-mllm1-answers.jsonl` → `text` field | all verification + integration |
| 3 | MLLM_2 pre-stored answer | `MLLMs_output/<Task>-mllm2-answers.jsonl` → `text` field | all verification + integration |
| 4 | MLLM_3 pre-stored answer | `MLLMs_output/<Task>-mllm3-answers.jsonl` → `text` field | all verification + integration |
| 5 | Classifier predictions (CONCH, MIZero, TITAN) | `classifier_outputs/*.jsonl` | EKV/Consensus only |
| 6 | Medical knowledge base | `medical_kb_structured/` (Chroma vector DB) | EKV/Fact only |

---

## Pipeline per Task Type

All four task types use the same full verification pipeline. The only differences are which MLLM keys each Expert Agent reads and what question semantics are passed to the verification agents.

```
┌──────────────────────────────────────────────────────────┐
│  Common full verification pipeline (all 4 task types)    │
│                                                          │
│  Expert Agent  (reads MLLM_1 … MLLM_k)                  │
│       │   fan-out (parallel)                             │
│  ┌────┴──────┬──────────────┐                            │
│  ▼           ▼              ▼                            │
│  ICV       EKV/Fact      EKV/Consensus                   │
│  Logic     Knowledge     Classifier                      │
│  Agent     Agent         Agent                           │
│  φ_l       φ_k           φ_c                             │
│  └────┬──────┴──────────────┘                            │
│       │   all 3 converge                                 │
│       ▼                                                  │
│  IntegrationAgent                                        │
│  φ_total → best MLLM_k* → refine → 3× review vote       │
│       │                                                  │
│       ▼                                                  │
│  refined_responses_<Task>.jsonl                          │
└──────────────────────────────────────────────────────────┘
```

### Task types

The system has four task types, each corresponding to one pipeline route:

| Task type | Route | Description |
|-----------|-------|-------------|
| **Morphology** | `morphology` | Morphological analysis — architectural patterns, tissue organization, structural and cellular features |
| **Diagnosis** | `diagnosis` | Pathological diagnosis — cancer classification, grading, and diagnostic conclusions |
| **Treatment** | `treatment` | Clinical management — TNM staging, subtype identification, prognosis, treatment planning |
| **Report** | `report` | Full structured pathology report generation from WSI findings |

The MLLM keys read per route:

| Route | MLLM keys read |
|-------|----------------|
| `morphology` | mllm_1, mllm_2, mllm_3 |
| `diagnosis` | mllm_1, mllm_2, mllm_3 |
| `treatment` | mllm_1, mllm_2, mllm_3 |
| `report` | mllm_1, mllm_2, mllm_3 |

> **Adding a new task type:** add an Expert Agent in `MLLM_agent.py`, register it in `MedicalAnalysisSystem._register_agents()`, add the keyword mapping to `TaskAllocationAgent`'s system prompt, add an entry to `ALL_TASKS` in `run_experiments.py`, and declare which `MLLM_PATHS` keys it reads. Model files are configured entirely in `Config.MLLM_PATHS`.

---

## Agent Details

### 1 · TaskAllocationAgent  *(TAM — Task Agent)*
`agent.py`

GPT-4o classifies the question and routes to the corresponding Expert Agent:

| Type | Keywords | Route |
|------|----------|-------|
| `morphology` | structure, pattern, appearance, cellular features | MorphologyAgent |
| `diagnosis` | diagnosis, disease, condition, finding | DiagnosisAgent |
| `treatment` | TNM, staging, treatment, prognosis, subtype, biomarker | TreatmentAgent |
| `report` | report, generate report, pathology report, summarize findings | ReportAgent |

---

### 2 · Expert Agents  *(TAM)*
`MLLM_agent.py`

Each Expert Agent reads the MLLM keys it needs from `Config.MLLM_PATHS`, wraps the answers in an `AnalysisTask`, and publishes to its own set of downstream topics. Adding a new task type = adding a new Expert Agent here.

| Agent | MLLM keys read | Publishes to |
|-------|----------------|-------------|
| **MorphologyAgent** | `mllm_1`, `mllm_2`, `mllm_3` | `consistency` + `verification` + `classifier_verification` + `integration` |
| **DiagnosisAgent** | `mllm_1`, `mllm_2`, `mllm_3` | `consistency` + `verification` + `classifier_verification` + `integration` |
| **TreatmentAgent** | `mllm_1`, `mllm_2`, `mllm_3` | `consistency` + `verification` + `classifier_verification` + `integration` |
| **ReportAgent** | `mllm_1`, `mllm_2`, `mllm_3` | `consistency` + `verification` + `classifier_verification` + `integration` |

---

### 3 · InternalConsistencyAgent  *(ICV — Logic Agent → φ_l)*
`InternalValidation.py`

Evaluates each MLLM answer independently using GPT-4o. Two sub-scores per answer:

| Score | Criteria |
|-------|----------|
| **CCS** (Content Consistency) | Descriptive coherence · Morphological correlation · Question-answer relevance |
| **RCS** (Reasoning Consistency) | Logical coherence · Conclusion-evidence alignment · Feature-diagnosis correlation |

```
φ_l = ConsistencyScore = (CCS + RCS) / 2        range [0, 1]
```

Output: `ConsistencyResult` with scores + issue list for each of MLLM_1 / MLLM_2 / MLLM_3

---

### 4 · KnowledgeVerificationAgent  *(EKV — Fact Agent → φ_k)*
`ExternalValidation.py`

Three steps per MLLM answer:

1. **Extract diagnosis** via GPT-4o
2. **Retrieve** top-3 WHO criteria chunks from Chroma vector DB
3. **Verify** — GPT-4o outputs matched / unmatched / incorrect rule lists

```
φ_k = KnowledgeScore = matched / (matched + unmatched) − 0.1 × incorrect    range [0, 1]
```

Output: `VerificationResult` with score + rule lists per MLLM

---

### 5 · ClassifierVerificationAgent  *(EKV — Consensus Agent → φ_c)*
`ExternalValidation.py`

Cross-checks MLLM diagnoses against three vision classifiers (CONCH, MIZero, TITAN):

```
MCS = mean(semantic_similarity_i × confidence_i)   # MLLM-classifier agreement
CIS = majority_label_count / total_classifiers      # inter-classifier agreement
φ_c = CVS = max(MCS × CIS, 0.01)
```

Output: `ClassifierVerificationResult` with MCS, CIS, CVS per MLLM

---

### 6 · IntegrationAgent  *(Summary Module — Path A)*
`IntegrationAgent.py`

Waits for all three verification results + the original `AnalysisTask`, then:

**Step 1 — Score & select**
```
φ_total = 0.2 × φ_l  +  0.3 × φ_k  +  0.5 × φ_c
```
MLLM with highest φ_total becomes base response MLLM_k*.

**Step 2 — Refine**  
GPT-4o edits MLLM_k* to remove inconsistencies, align with classifier predictions (confidence > 0.7), add missing diagnostic features.

**Step 3 — Iterative review**  
GPT-4o proposes an enhanced version incorporating non-conflicting details from the other MLLMs.  
3 Reasoning Agent instances vote:
- **≥ 3 approvals** → accept
- **< 3** → revise based on feedback, repeat up to 3 rounds

---

### 7 · AnswerSelectionAgent + VerificationPanelAgent  *(retained, not on primary path)*
`agent.py`

These agents implement an alternative selection pipeline: GPT-4o directly selects the best answer from MLLM_1–4, then a multi-reviewer panel votes approve/reject (≥ 2 approvals to accept, up to 3 revision rounds). They are retained in the codebase; all task types currently go through `IntegrationAgent` for scoring and refinement.

---

## Score Summary

| Score | Paper notation | Formula | Weight in φ_total |
|-------|---------------|---------|-------------------|
| ConsistencyScore | φ_l | (CCS + RCS) / 2 | **0.2** |
| KnowledgeScore | φ_k | matched/(matched+unmatched) − 0.1×incorrect | **0.3** |
| CVS | φ_c | max(MCS × CIS, 0.01) | **0.5** |
| MCS | ≈ φ_a | mean(similarity × confidence) | — |
| CIS | ≈ φ_b | majority_count / total_classifiers | — |

---

## Knowledge Base

Built once from WHO Classification of Tumours books (`.docx`) using LangChain + Chroma.

```
WHO books (.docx) + JSON knowledge docs
          │
          ▼
   DocxProcessor → text chunks (1500 tokens, 200 overlap)
          │
          ▼
   OpenAIEmbeddings → Chroma vector store (medical_kb_structured/)
```

- **Retrieval:** top-3 nearest chunks per query
- **QA chain:** GPT-4o via `ConversationalRetrievalChain`

```bash
python knowledge_base_demo/build_kb.py   # build
python knowledge_base_demo/kb_demo.py    # query interactively
```

---

## Code Structure

```
open_source/
│
├── requirements.txt            # Python dependencies
├── run_experiments.py          # Batch runner — iterates all task categories
├── run_demo.py                 # Demo runner — runs 1 case per task to demo_output/
├── MedicalAnalysisSystem.py    # System class + per-case analysis loop
├── config.py                   # All paths + MLLM_PATHS k-v; configure_task() switches tasks
├── model_client.py             # API key / base URL (env vars: OPENAI_API_KEY, OPENAI_BASE_URL)
│
├── agent.py                    # TaskAllocationAgent (TAM)
│                               # AnswerSelectionAgent, VerificationPanelAgent (retained)
├── MLLM_agent.py               # MorphologyAgent, DiagnosisAgent, TreatmentAgent, ReportAgent (TAM)
├── InternalValidation.py       # InternalConsistencyAgent (ICV / Logic Agent)
├── ExternalValidation.py       # KnowledgeVerificationAgent (EKV / Fact Agent)
│                               # ClassifierVerificationAgent (EKV / Consensus Agent)
├── IntegrationAgent.py         # IntegrationAgent (Summary Module)
│
├── base_models.py              # Pydantic message/result models
├── ScoreCalculator.py          # φ_l, φ_k, φ_c, φ_total implementations
├── ResponseReader.py           # JSONL reader for MLLM answer files
├── ClassifierResultsReader.py  # JSONL reader for classifier prediction files
├── knowledge_base.py           # MedicalKnowledgeBuild + MedicalKnowledgeBase
│
└── knowledge_base_demo/
    ├── build_kb.py             # Build Chroma DB from WHO books + JSON docs
    └── kb_demo.py              # Query the KB interactively
```

---

## Data Layout

```
autogen/                                (project root, one level above open_source/)
│
├── questions/                          Input questions — one file per task type
│   ├── Morphology-questions.jsonl          {"question_id", "prompt"}
│   ├── Diagnosis-questions.jsonl
│   ├── Treatment-questions.jsonl
│   └── Report-questions.jsonl
│
├── MLLMs_output/                       Pre-stored MLLM answers — one file per model
│   ├── mllm1-answers.jsonl                 MLLM_1 answers — shared across all tasks
│   ├── mllm2-answers.jsonl                 MLLM_2 answers — shared across all tasks
│   ├── Morphology-mllm3-answers.jsonl      MLLM_3 answers for Morphology task
│   ├── Diagnosis-mllm3-answers.jsonl       … Diagnosis
│   ├── Treatment-mllm3-answers.jsonl       … Treatment
│   └── Report-mllm3-answers.jsonl          … Report
│
├── classifier_outputs/                 Vision classifier predictions (per slide)
│   ├── Conch.jsonl                         {"question_id", "label", "confidence"}
│   ├── MIZero.jsonl
│   └── TITAN.jsonl
│
├── medical_kb_structured/              Chroma vector DB (built by build_kb.py)
│
├── output/                             Refined answer outputs
│   └── refined_responses_<Task>.jsonl      {"question_id", "text"}
│
└── demo_output/                        Demo run outputs (1 case per task type)
```

---

## Message Flow (autogen_core topics)

```
── All task types share the same fan-out pattern ─────────────────────
task_allocation              →  morphology / diagnosis / treatment / report
<expert_agent>               →  consistency + verification
                                + classifier_verification + integration
consistency                  →  integration
verification                 →  integration
classifier_verification      →  integration
integration                  →  final_result
```

Each topic maps to exactly one registered agent. `session_id` (= `question_id`) is carried in `TopicId.source` so agents correlate messages for the same case.
