# src/fitness.py
import pandas as pd
from typing import Dict, List


def evaluate(schedule: List[int], jobs: List[Dict[str, int]]) -> int:
    """
    Compute total weighted tardiness for a given schedule.

    Total Weighted Tardiness = sum(weight_j * max(0, completion_j - deadline_j))

    Args:
        schedule: ordered list of job IDs
        jobs: list of job dicts with processing_time, deadline, weight

    Returns:
        integer total weighted tardiness (lower is better)
    """
    current_time = 0
    total = 0
    for job_id in schedule:
        job = jobs[job_id]
        current_time += job["processing_time"]
        tardiness = max(0, current_time - job["deadline"])
        total += job["weight"] * tardiness
    return int(total)


def schedule_details(schedule: List[int], jobs: List[Dict[str, int]]) -> pd.DataFrame:
    """
    Return a detailed breakdown of a schedule — start, completion,
    tardiness, weighted tardiness, and late flag for each job.

    Args:
        schedule: ordered list of job IDs
        jobs: list of job dicts

    Returns:
        DataFrame with one row per job in schedule order
    """
    rows = []
    current_time = 0
    for position, job_id in enumerate(schedule):
        job = jobs[job_id]
        start = current_time
        current_time += job["processing_time"]
        tardiness = max(0, current_time - job["deadline"])
        rows.append({
            "position": position,
            "job_id": job_id,
            "processing_time": job["processing_time"],
            "deadline": job["deadline"],
            "weight": job["weight"],
            "start": start,
            "completion": current_time,
            "tardiness": tardiness,
            "weighted_tardiness": tardiness * job["weight"],
            "is_late": tardiness > 0,
        })
    return pd.DataFrame(rows)


def summarize_late_jobs(schedule: List[int], jobs: List[Dict[str, int]]) -> List[Dict]:
    """
    Return a list of dicts describing late jobs — used to build LLM prompts.

    Args:
        schedule: ordered list of job IDs
        jobs: list of job dicts

    Returns:
        list of dicts with job_id, tardiness, weight, weighted_tardiness, position
    """
    df = schedule_details(schedule, jobs)
    late = df[df["is_late"]].copy()
    late["position"] = late["job_id"].apply(lambda jid: schedule.index(jid))
    return late[["job_id", "tardiness", "weight", "weighted_tardiness", "position"]].to_dict("records")
