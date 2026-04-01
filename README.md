# LLM-Guided Tree of Thoughts Scheduler Job Sequencing Optimization 🗓️

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-DeepSeek--R1:7b-black?logo=ollama&logoColor=white)](https://ollama.com/)
[![Colab](https://img.shields.io/badge/Notebook-Google%20Colab%20T4-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Research-F59E0B)](https://github.com/Ayoub-Elkhaiari)

LLM-guided **Tree of Thoughts beam search** for the **Single Machine Total Weighted Tardiness** problem. A locally-running 7B reasoning model proposes targeted job swaps inside an iterative search loop — outperforming all classical scheduling heuristics by up to **28.8%** on the OR-Library `wt40` benchmark.

---

## The Core Idea

> *Can a locally-running LLM reason about scheduling constraints well enough to outperform classical dispatching rules — without any training on scheduling data?*

Classical methods like EDF, SPT, and WSPT are fast and deterministic but apply rigid single-rule logic. This project uses **DeepSeek-R1:7b** not as a solver, but as a **guided neighborhood generator** inside a beam search loop. The LLM reads the current schedule, identifies which jobs are late and how costly their tardiness is, and proposes targeted moves. Python applies the moves and evaluates them with the exact objective function. The best candidates survive into the next round.

---

## Results

### Evaluated on OR-Library `wt40` instance 0 (40 jobs)

| Method | Total Weighted Tardiness | vs LLM Tree Search |
|---|---|---|
| **LLM Tree Search** | **1131** ⭐ | — |
| EDF (Earliest Deadline First) | 1588 | +40.4% worse |
| WSPT (Weighted Shortest Processing Time) | 3066 | +171.1% worse |
| SPT (Shortest Processing Time) | 7755 | +585.8% worse |
| Random | 10476 | +826.3% worse |

### Fitness progression over search depth

```
Depth 0 →  1588   (EDF seed — best classical baseline)
Depth 1 →  1379   (-13.2%  first LLM-guided improvement)
Depth 2 →  1343   (-2.6%   further refinement)
Depth 3 →  1131   (-15.8%  largest single-depth gain)
Depth 4 →  1131   (converged)
Depth 5 →  1131   (stable)
```

### Fitness curve vs EDF baseline

The blue curve shows the LLM tree search improving at every depth. The red dashed line is EDF — the best classical method. The LLM beats it from depth 1 onwards and converges 28.8% below it at depth 3.

<img width="790" height="490" alt="immagine" src="https://github.com/user-attachments/assets/5d65c68f-d4b6-40e8-9ccb-de0617af1c1c" />


### Method comparison

<img width="1389" height="1110" alt="immagine" src="https://github.com/user-attachments/assets/7c78a5e9-72ea-4047-83e9-a77002fcb0dc" />


### Gantt chart — best schedule found

Almost entirely green (on-time jobs). Only 4 jobs at the very end are late all unavoidable given total processing load. The LLM learned to front-load short, high-weight jobs to keep the timeline tight.

<img width="1389" height="1110" alt="immagine" src="https://github.com/user-attachments/assets/88f160d7-d91e-45b8-b02a-59b5449efc18" />


---

## How It Works

```
Input: 40 jobs — each with processing_time, deadline, weight
          │
          ▼
Phase 1 — Classical Baselines
  Random (10476) → SPT (7755) → WSPT (3066) → EDF (1588) ← seed
          │
          ▼
Phase 2 — Tree of Thoughts Search (depth 0 → 5)

  For each depth:
    LLM reads: current schedule + late jobs + fitness
    LLM proposes: 3 swaps [[job_id, new_pos], ...]
    Python applies swaps → 3 child schedules
    Exact fitness evaluation
    Keep top-3 beam → next depth
          │
          ▼
Phase 3 — Convergence at depth 3: fitness 1131
```

### Why swaps instead of full permutations?

Asking a 7B model to output a full 40-job permutation reliably produces duplicates and omissions. Asking for 3 targeted swaps `[[job_id, new_pos], ...]` is a simpler task — the model only decides *which job* to move and *where*. Python handles permutation manipulation and validates every result.

### Why the LLM adds value over random search

A random mutation swaps any two jobs blindly. The LLM specifically targets **late high-weight jobs** and moves them earlier. This is directional intelligence — it understands that a job with weight 10 that is 77 units late costs 770 penalty points and addresses the highest-cost problems first.

### Fallback mechanism

When the LLM returns malformed output the parser falls back to random swap mutations. The search never stops. This happened several times during the run — yet the search still converged to 1131 because successful LLM calls were directional enough to compensate.

---

## Project Structure

```
LLM_Guided_Job_Sequencing_Optimization/
├── Modular/                        # Production-ready Python package
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data.py                 # OR-Library loader + synthetic fallback
│   │   ├── fitness.py              # evaluate(), schedule_details()
│   │   ├── baselines.py            # Random, EDF, SPT, WSPT
│   │   ├── llm.py                  # prompt builder, swap parser, propose_schedules()
│   │   ├── search.py               # tree_of_thoughts_search()
│   │   └── visualize.py            # fitness curve, Gantt, bar chart
│   ├── main.py                     # CLI entry point
│   └── requirements.txt
│
└── Notebook/                       # Interactive Colab walkthrough
    └── llm_guided_tree_search_job_scheduling.ipynb
```

---

## Installation (Modular)

```bash
git clone https://github.com/Ayoub-Elkhaiari/LLM_Guided_Job_Sequencing_Optimization.git
cd LLM_Guided_Job_Sequencing_Optimization/Modular
pip install -r requirements.txt
```

Install and start Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:7b
ollama serve
```

---

## Usage (Modular)

### Run with defaults
```bash
python main.py
```

### Custom depth and beam width
```bash
python main.py --depth 8 --beam 5
```

### Different OR-Library instance
```bash
python main.py --instance 10
```

### No plots (headless)
```bash
python main.py --no-plots --save-results results/run.csv
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--instance` | `0` | OR-Library wt40 instance (0–124) |
| `--depth` | `5` | Maximum search depth |
| `--beam` | `3` | Beam width (top-K per depth) |
| `--model` | `deepseek-r1:7b` | Ollama model name |
| `--no-plots` | off | Skip matplotlib plots |
| `--synthetic` | off | Use synthetic 15-job dataset |
| `--save-results` | `results/results.csv` | Output CSV path |

### Use as a library

```python
from langchain_ollama import OllamaLLM
from src.data import load_jobs
from src.baselines import get_all_baselines
from src.fitness import evaluate
from src.search import tree_of_thoughts_search
from src.visualize import plot_fitness_progression

jobs, info = load_jobs(instance_idx=0)
baselines = get_all_baselines(jobs)
baseline_scores = {name: evaluate(s, jobs) for name, s in baselines.items()}

model = OllamaLLM(model="deepseek-r1:7b", temperature=0.7)
best_schedule, best_fitness, history = tree_of_thoughts_search(
    jobs=jobs,
    model=model,
    max_depth=5,
    beam_width=3,
)

print(f"Best fitness: {best_fitness}")
plot_fitness_progression(history, baseline_scores["EDF"], "EDF")
```

---

## Notebook (Google Colab T4)

The `Notebook/` folder contains a complete self-contained Colab notebook that:
- Installs Ollama and pulls DeepSeek-R1:7b automatically
- Loads the OR-Library benchmark with synthetic fallback
- Runs all classical baselines
- Runs the full Tree of Thoughts search
- Produces all 3 visualizations inline

Open in Colab and run top to bottom — no local setup required.

---

## Limitations & Future Work

- **Scale**: tested on 40 jobs — scaling to 100+ requires prompt compression
- **Multi-machine**: single machine only — Job Shop extension planned
- **Statistical rigor**: single-run results — multiple seeds with std needed for a paper
- **Optimal gap**: comparison against branch-and-bound optimal not yet included
- **Instance coverage**: testing across all 125 wt40 instances to measure variance

---

## Research Context

This project explores a question central to automated optimization:

> *Can an LLM's qualitative reasoning about constraint satisfaction guide a combinatorial search more efficiently than blind heuristics?*

The answer here is yes for this problem, on this instance, with this model. The LLM does not replace exact evaluation. It acts as a **proposal engine** that understands penalty structure and directs search toward high-impact moves. The result is a hybrid system where language reasoning and exact optimization work together.

This connects to emerging work on LLMs as meta-heuristics, algorithm selection, and learned landscape features — topics at the frontier of automated algorithm design.

---

## Citation

```bibtex
@misc{elkhaiari2025llmscheduler,
  author    = {EL KHAIARI, Ayoub},
  title     = {LLM-Guided Job Sequencing Optimization via Tree of Thoughts Beam Search},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/Ayoub-Elkhaiari/LLM_Guided_Job_Sequencing_Optimization}
}
```

---

## Author

**Ayoub EL KHAIARI** — AI Researcher
[Medium](https://medium.com/@elkhaiariayoub) · [GitHub](https://github.com/Ayoub-Elkhaiari)

---

*"The LLM does not solve the problem. It tells you where to look."*
