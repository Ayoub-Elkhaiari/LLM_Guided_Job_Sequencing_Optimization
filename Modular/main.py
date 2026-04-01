# main.py
"""
LLM-Guided Tree of Thoughts for Single Machine Scheduling
==========================================================
Entry point for running the full experiment from the command line.

Usage:
    python main.py                          # default: wt40 instance 0
    python main.py --instance 3             # use OR-Library instance 3
    python main.py --depth 8 --beam 5       # deeper search
    python main.py --no-plots               # skip visualizations
    python main.py --synthetic              # use synthetic dataset

Requirements:
    - Ollama running locally with deepseek-r1:7b pulled
    - pip install langchain-ollama langchain matplotlib pandas numpy requests
"""

import argparse
import time
import pandas as pd
from langchain_ollama import OllamaLLM

from src.data import load_jobs, generate_synthetic_jobs
from src.fitness import evaluate
from src.baselines import get_all_baselines
from src.search import tree_of_thoughts_search
from src.visualize import plot_fitness_progression, plot_gantt, plot_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-guided scheduling optimizer")
    parser.add_argument("--instance", type=int, default=0, help="OR-Library wt40 instance index (0-124)")
    parser.add_argument("--depth", type=int, default=5, help="Maximum tree search depth")
    parser.add_argument("--beam", type=int, default=3, help="Beam width (top-K candidates per depth)")
    parser.add_argument("--model", type=str, default="deepseek-r1:7b", help="Ollama model name")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualizations")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset instead of OR-Library")
    parser.add_argument("--save-results", type=str, default="results/results.csv", help="Path to save results CSV")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("\n── Loading dataset ─────────────────────────────────────────────")
    if args.synthetic:
        jobs = generate_synthetic_jobs(n=15, seed=42)
        print(f"Using synthetic dataset: {len(jobs)} jobs")
    else:
        jobs, info = load_jobs(instance_idx=args.instance)
        print(f"Dataset: {info['source']} | Instance: {args.instance} | Jobs: {info['n_jobs']}")

    # ── 2. Classical baselines ───────────────────────────────────────────────
    print("\n── Classical baselines ─────────────────────────────────────────")
    baseline_schedules = get_all_baselines(jobs)
    baseline_scores = {name: evaluate(s, jobs) for name, s in baseline_schedules.items()}
    for name, score in sorted(baseline_scores.items(), key=lambda x: x[1]):
        print(f"  {name:<8} → {score}")
    best_baseline = min(baseline_scores, key=baseline_scores.get)

    # ── 3. LLM setup ────────────────────────────────────────────────────────
    print(f"\n── LLM setup ({args.model}) ─────────────────────────────────────")
    model = OllamaLLM(model=args.model, temperature=0.7)
    print("Model ready.")

    # ── 4. Tree of Thoughts search ───────────────────────────────────────────
    print("\n── Tree of Thoughts search ─────────────────────────────────────")
    t0 = time.time()
    llm_schedule, llm_score, history = tree_of_thoughts_search(
        jobs=jobs,
        model=model,
        max_depth=args.depth,
        beam_width=args.beam,
        verbose=True,
    )
    llm_runtime = time.time() - t0
    print(f"\nSearch complete in {llm_runtime:.1f}s")

    # ── 5. Results table ─────────────────────────────────────────────────────
    print("\n── Results ─────────────────────────────────────────────────────")
    rows = []
    for name, s in baseline_schedules.items():
        rows.append({"Method": name, "Total Weighted Tardiness": baseline_scores[name], "Runtime (s)": 0.0})
    rows.append({"Method": "LLM Tree Search", "Total Weighted Tardiness": llm_score, "Runtime (s)": round(llm_runtime, 2)})

    results_df = pd.DataFrame(rows).sort_values("Total Weighted Tardiness").reset_index(drop=True)
    print(results_df.to_string(index=False))

    improvement = (baseline_scores[best_baseline] - llm_score) / baseline_scores[best_baseline] * 100
    print(f"\nLLM improvement over {best_baseline}: {improvement:.1f}%")

    # ── 6. Save results ──────────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    results_df.to_csv(args.save_results, index=False)
    print(f"Results saved to {args.save_results}")

    # ── 7. Visualizations ────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n── Visualizations ──────────────────────────────────────────────")
        plot_fitness_progression(history, baseline_scores[best_baseline], best_baseline)
        plot_gantt(llm_schedule, jobs, title="LLM Tree Search")
        plot_comparison(results_df)


if __name__ == "__main__":
    main()
