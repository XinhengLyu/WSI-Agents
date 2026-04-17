# 🤖 WSI-Agents: A Collaborative Multi-Agent System for Multi-Modal Whole Slide Image Analysis

[![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025-blue)](https://conferences.miccai.org/2025/en/default.asp)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://drive.google.com/file/d/1T8aTBL_-JIZpKoRbvvYmoj2JisXssQMr/view)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**WSI-Agents** is a novel collaborative multi-agent system for whole slide image analysis that bridges the gap between accuracy and versatility in digital pathology through specialized agents and robust verification mechanisms.

🎉 **Accepted at MICCAI 2025**

---

## 📂 Resources

**Framework**: Built on [AutoGen](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/quickstart.html) for multi-agent orchestration.

**External Knowledge Base**: Knowledge base querying is implemented with [LangChain](https://www.langchain.com/) + Chroma vector store, backed by WHO Classification of Tumours books.

> **Note**: The pre-built knowledge base cannot be released due to copyright restrictions on the WHO Classification of Tumours series. You can build your own using the provided `knowledge_base_demo/build_kb.py` script — see [Building the Knowledge Base](#building-the-knowledge-base) below.

**Text Extraction**: Text from the WHO books is extracted using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s PP-StructureV2 layout recovery feature, which preserves paragraph structure across multi-column pathology book layouts. The extracted text is then split into overlapping chunks and indexed into a Chroma vector store via OpenAI embeddings.

---

## 🏗️ System Overview

![WSI-Agents Workflow](static/image/fig1.png)

WSI-Agents employs a collaborative multi-agent approach to address the accuracy-versatility trade-off in WSI analysis. The system integrates specialized expert agents with comprehensive verification mechanisms to ensure clinical accuracy while maintaining multi-task capabilities.

![WSI-Agents Architecture](static/image/wsi-agents.png)

The architecture consists of three core components:
- **Task Allocation Module** — classifies each question and routes it to the appropriate expert agent (Morphology, Diagnosis, Treatment, or Report)
- **Verification Mechanisms** — internal consistency checking (ICV) combined with external knowledge validation (EKV) via a medical knowledge base and vision classifier predictions
- **Summary Module** — scores all MLLM responses, selects the best, and iteratively refines the final answer

---

## 🎯 Key Innovations

- **Multi-Agent Collaboration**: Specialized agents for morphology analysis, diagnosis, treatment planning, and report generation
- **Dual Verification**: Internal consistency checking combined with external knowledge and classifier validation
- **Configurable MLLM Pool**: Models are defined as a key–value mapping in `config.py` — swapping a model requires changing one file path
- **Knowledge Integration**: Leverages WHO pathology knowledge bases and WSI foundation model predictions (CONCH, MIZero, TITAN)

---

> **Note**: The code is currently being organized and will be updated soon.

## 🛠️ Installation

```bash
git clone https://github.com/XinhengLyu/WSI-Agents
cd WSI_Agents

conda create -n WSI_Agents python=3.10 -y
conda activate WSI_Agents

pip install -r requirements.txt
```

Set API credentials:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## 📚 Building the Knowledge Base

The pre-built knowledge base is not included in this repository due to copyright restrictions on the WHO Classification of Tumours books. To build your own:

1. **Extract text** from the WHO Classification books using PaddleOCR's PP-StructureV2 layout recovery, which handles multi-column layouts and preserves paragraph structure:

   ```bash
   # Install PaddleOCR
   pip install paddlepaddle paddleocr

   # Run layout recovery on each book (outputs recovered .txt / .docx files)
   paddleocr --image_dir /path/to/book_pages --type structure --recovery True
   ```

2. **Build the Chroma vector store** from the extracted `.docx` files:

   ```bash
   # Place extracted .docx files in knowledge_base_demo/who_books/
   python knowledge_base_demo/build_kb.py
   # Outputs to: ../medical_kb_structured/
   ```

You can use any pathology reference text you have legal access to — the build script accepts any `.docx` files.

---

## 📊 Data Preparation

Questions and MLLM answers are stored in separate files:

```
autogen/
├── questions/                      One file per task type {"question_id", "prompt"}
│   ├── Morphology-questions.jsonl
│   ├── Diagnosis-questions.jsonl
│   ├── Treatment-questions.jsonl
│   └── Report-questions.jsonl
│
├── MLLMs_output/                   One file per model, each covers all question IDs {"question_id", "text"}
│   ├── modelA-answers.jsonl
│   ├── modelB-answers.jsonl
│   ├── modelC-answers.jsonl
│   └── ...                         (add as many models as needed)
│
└── classifier_outputs/             {"question_id", "label", "confidence"}
    ├── Conch.jsonl
    ├── MIZero.jsonl
    └── TITAN.jsonl
```

Configure all model file paths in `config.py`, then specify which three models each task uses:

```python
# Register all available model answer files
MLLM_PATHS = {
    "mllm_1": os.path.join(MLLM_OUTPUT_DIR, "modelA-answers.jsonl"),
    "mllm_2": os.path.join(MLLM_OUTPUT_DIR, "modelB-answers.jsonl"),
    "mllm_3": os.path.join(MLLM_OUTPUT_DIR, "modelC-answers.jsonl"),
    # add more models here
}

# Select which three models each task reads
TASK_MLLM_KEYS = {
    "Morphology": ["mllm_1", "mllm_2", "mllm_3"],
    "Diagnosis":  ["mllm_1", "mllm_2", "mllm_3"],
    "Treatment":  ["mllm_1", "mllm_2", "mllm_3"],
    "Report":     ["mllm_1", "mllm_2", "mllm_3"],
}
```

---

## 🚀 Usage

```bash
# Run all task types
python run_experiments.py

# Run a demo (1 case per task type, outputs to demo_output/)
python run_demo.py
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{lyu2025wsi,
  title={Wsi-agents: A collaborative multi-agent system for multi-modal whole slide image analysis},
  author={Lyu, Xinheng and Liang, Yuci and Chen, Wenting and Ding, Meidan and Yang, Jiaqi and Huang, Guolin and Zhang, Daokun and He, Xiangjian and Shen, Linlin},
  journal={arXiv preprint arXiv:2507.14680},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
