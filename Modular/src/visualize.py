# src/visualize.py
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from typing import Dict, List

from src.fitness import schedule_details


def plot_fitness_progression(history: List[Dict], baseline_fitness: int, baseline_name: str = "EDF") -> None:
    """
    Plot LLM tree search fitness over search depth.
    Shows the improvement curve alongside the best classical baseline.

    Args:
        history: list of dicts from tree_of_thoughts_search (depth, best_fitness)
        baseline_fitness: score of the best classical baseline for reference line
        baseline_name: name of the baseline method for the legend label
    """
    df = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(df["depth"], df["best_fitness"], marker="o", linewidth=2, color="#185FA5", label="LLM Tree Search")
    plt.axhline(
        y=baseline_fitness,
        color="#E24B4A",
        linestyle="--",
        linewidth=1.2,
        label=f"Best baseline ({baseline_name}): {baseline_fitness}",
    )
    plt.title("LLM Tree Search — Fitness Progression")
    plt.xlabel("Depth (iteration)")
    plt.ylabel("Best Total Weighted Tardiness")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_gantt(schedule: List[int], jobs: List[Dict], title: str = "Best Schedule") -> None:
    """
    Plot a Gantt chart for a schedule.
    Green bars = on-time jobs, red bars = late jobs.
    Vertical dashed lines mark each job's deadline.

    Args:
        schedule: ordered list of job IDs
        jobs: list of job dicts
        title: chart title
    """
    df = schedule_details(schedule, jobs)
    fig_height = max(6, len(df) * 0.28)
    plt.figure(figsize=(14, fig_height))

    for idx, row in df.iterrows():
        color = "tab:green" if not row["is_late"] else "tab:red"
        plt.barh(idx, row["processing_time"], left=row["start"], color=color, edgecolor="black", alpha=0.85)
        plt.axvline(row["deadline"], color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        plt.text(
            row["start"] + row["processing_time"] / 2,
            idx,
            f"J{int(row['job_id'])}",
            ha="center", va="center",
            fontsize=7, color="white", fontweight="bold",
        )

    plt.yticks(range(len(df)), [f"Pos {i}" for i in range(len(df))])
    plt.xlabel("Time")
    plt.title(f"Gantt Chart — {title}")
    plt.legend(handles=[
        Patch(facecolor="tab:green", label="On time"),
        Patch(facecolor="tab:red", label="Late"),
    ])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_comparison(results_df: pd.DataFrame) -> None:
    """
    Bar chart comparing all methods on total weighted tardiness.
    LLM Tree Search bar is highlighted in orange; all others in blue.

    Args:
        results_df: DataFrame with columns Method and Total Weighted Tardiness
    """
    plot_df = results_df.sort_values("Total Weighted Tardiness").copy()
    colors = [
        "tab:orange" if m == "LLM Tree Search" else "tab:blue"
        for m in plot_df["Method"]
    ]
    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["Method"], plot_df["Total Weighted Tardiness"], color=colors, edgecolor="black")
    plt.title("Method Comparison — Total Weighted Tardiness (lower is better)")
    plt.ylabel("Total Weighted Tardiness")
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
