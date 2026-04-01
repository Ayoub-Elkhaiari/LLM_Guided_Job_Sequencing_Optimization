# src/data.py
import re
import numpy as np
import requests
from typing import Dict, List, Tuple

ORLIB_URLS = [
    "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/wt40.txt",
    "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/wt40.txt",
]


def generate_synthetic_jobs(n: int = 15, seed: int = 42) -> List[Dict[str, int]]:
    """Generate synthetic jobs for testing when OR-Library is unavailable."""
    rng = np.random.default_rng(seed)
    return [
        {
            "id": i,
            "processing_time": int(rng.integers(1, 20)),
            "deadline": int(rng.integers(20, 100)),
            "weight": int(rng.integers(1, 10)),
        }
        for i in range(n)
    ]


def parse_wt40_text(text: str, instance_idx: int = 0, n_jobs: int = 40) -> List[Dict[str, int]]:
    """Parse the OR-Library wt40 benchmark file format."""
    numbers = [int(x) for x in re.findall(r"-?\d+", text)]
    block_size = 3 * n_jobs
    n_instances = len(numbers) // block_size

    if instance_idx < 0 or instance_idx >= n_instances:
        raise IndexError(f"instance_idx={instance_idx} out of range (max {n_instances - 1})")

    start = instance_idx * block_size
    processing_times = numbers[start: start + n_jobs]
    weights = numbers[start + n_jobs: start + 2 * n_jobs]
    deadlines = numbers[start + 2 * n_jobs: start + 3 * n_jobs]

    return [
        {
            "id": i,
            "processing_time": processing_times[i],
            "deadline": deadlines[i],
            "weight": weights[i],
        }
        for i in range(n_jobs)
    ]


def load_jobs(instance_idx: int = 0) -> Tuple[List[Dict[str, int]], Dict]:
    """
    Load jobs from OR-Library wt40 benchmark.
    Falls back to synthetic dataset if download fails.

    Args:
        instance_idx: which instance to load (0-124 for wt40)

    Returns:
        jobs: list of job dicts with id, processing_time, deadline, weight
        info: metadata dict about the dataset source
    """
    last_error = None
    for url in ORLIB_URLS:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            jobs = parse_wt40_text(r.text, instance_idx)
            return jobs, {
                "source": "OR-Library wt40",
                "url": url,
                "instance_idx": instance_idx,
                "n_jobs": len(jobs),
            }
        except Exception as e:
            last_error = e

    jobs = generate_synthetic_jobs(n=15, seed=42)
    return jobs, {
        "source": "Synthetic fallback",
        "url": None,
        "instance_idx": None,
        "n_jobs": len(jobs),
        "warning": str(last_error) if last_error else "Unknown download failure",
    }
