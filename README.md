

# Question Generation for Real Classrooms

This repository contains (1) a new AIRC dataset of educator created multiple-choice questions for lectures from real-world classrooms, and (2) a context selection and rewriting pipeline for video-based educational QG.

The pipeline supports models using OpenAI api and TogetherAI api.

## Updates
[2025.04.27] Uploaded arxiv paper.
[2025.04.15] Uploaded quiz generation data for two courses: `LLM-Frontier`, and `DL-Intro`.

## 🚀 Overview

The main script, `run_chatgpt_pipeline.py`, is executed via a Bash script that sweeps over combinations of:
- **Context choices** (e.g., `CoTT`, `DirectT`, `CoTV`, `DirectV`, `Full`, `RuleT3`, etc.)
- **Rewrite choices** (`Yes` or `No`)

The outputs are saved per configuration, allowing for easy analysis and comparison of different strategies.

## 🧠 Context Choices

The pipeline supports a variety of context formats:

| Context Format | Description |
|----------------|-------------|
| `CoTT`, `DirectT` | Text-based context, Chain-of-Thought (CoT) or Direct |
| `CoTV`, `DirectV` | Visual-context-enhanced, with CoT or Direct |
| `CoTMM`, `DirectMM` | Multi-modal inputs combining text and visuals |
| `Full` | Full original context without trimming or rewriting |
| `RuleT3`, `RuleV3` | Rule-based context formats, V and T variants |

## ✍️ Rewrite Choices

- `Yes`: Apply rewriting strategies to enhance clarity and conciseness of the input context
- `No`: Use raw extracted context

## 📂 Directory Structure

```bash
.
├── run_pipeline.sh             # Bash script for running all config combinations
├── run_chatgpt_pipeline.py    # Main pipeline script
├── ../data/                   # Data directory (outside repo)
│   ├── LLM-Frontier/
│   └── MIT-DL/
├── ../out/                    # Output directory (auto-generated)
```

Each run outputs to:

```
../out/<out_folder>/<model_name>/<data_split>/<context>_<rewrite>/<run_id>/
```

## 🔧 Configuration

You can customize the following parameters inside `run_pipeline.sh`:

```bash
model_name=gpt-4o-mini        # or other HuggingFace / OpenAI model names
max_context_length=4000
max_output_length=800
temperature=0.1
seed=42
run_id=0415-1
out_folder=ND-LLM
data_split=ND-LLM
device=cpu                    # or cuda if available
api_key=your_api_key          # required for OpenAI/Together APIs
```

## 🛠️ Dependencies

Make sure the following dependencies are installed:

- Python 3.10+
- `transformers`
- `openai`
- `together`
- `tqdm`

Install via:

```bash
conda create -n coser python=3.10
conda activate coser
pip install -r requirements.txt
```



## ▶️ Running the Pipeline

```bash
cd code/
bash run_COSER_pipeline.sh
```

This will execute all combinations of context and rewrite strategies and save the output in a structured format.

