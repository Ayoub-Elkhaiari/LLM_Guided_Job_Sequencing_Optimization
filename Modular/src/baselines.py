# src/baselines.py
import random
from typing import Dict, List


def random_schedule(jobs: List[Dict[str, int]], seed: int = 42) -> List[int]:
    """
    Random job ordering.
    Baseline: no intelligence applied.

    Args:
        jobs: list of job dicts
        seed: random seed for reproducibility

    Returns:
        list of job IDs in random order
    """
    rng = random.Random(seed)
    schedule = [j["id"] for j in jobs]
    rng.shuffle(schedule)
    return schedule


def edf_schedule(jobs: List[Dict[str, int]]) -> List[int]:
    """
    Earliest Deadline First (EDF).
    Sort jobs by deadline ascending — tightest deadline goes first.
    Best classical baseline for deadline-constrained scheduling.

    Args:
        jobs: list of job dicts

    Returns:
        list of job IDs sorted by deadline
    """
    return [
        j["id"]
        for j in sorted(jobs, key=lambda x: (x["deadline"], x["processing_time"], -x["weight"], x["id"]))
    ]


def spt_schedule(jobs: List[Dict[str, int]]) -> List[int]:
    """
    Shortest Processing Time (SPT).
    Run fastest jobs first to minimize average completion time.
    Ignores deadlines and weights.

    Args:
        jobs: list of job dicts

    Returns:
        list of job IDs sorted by processing time
    """
    return [
        j["id"]
        for j in sorted(jobs, key=lambda x: (x["processing_time"], x["deadline"], x["id"]))
    ]


def wspt_schedule(jobs: List[Dict[str, int]]) -> List[int]:
    """
    Weighted Shortest Processing Time (WSPT).
    Sort by processing_time / weight — prioritizes fast and important jobs.
    Optimal for minimizing total weighted completion time (no deadlines).

    Args:
        jobs: list of job dicts

    Returns:
        list of job IDs sorted by processing_time / weight ratio
    """
    return [
        j["id"]
        for j in sorted(jobs, key=lambda x: (x["processing_time"] / x["weight"], x["deadline"], x["id"]))
    ]


def get_all_baselines(jobs: List[Dict[str, int]]) -> Dict[str, List[int]]:
    """
    Compute all classical baseline schedules at once.

    Returns:
        dict mapping method name to schedule (list of job IDs)
    """
    return {
        "Random": random_schedule(jobs, seed=42),
        "EDF": edf_schedule(jobs),
        "SPT": spt_schedule(jobs),
        "WSPT": wspt_schedule(jobs),
    }
