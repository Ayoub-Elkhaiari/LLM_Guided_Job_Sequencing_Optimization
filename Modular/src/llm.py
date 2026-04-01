# src/llm.py
import json
import random
import re
from typing import Dict, List

from src.fitness import evaluate, summarize_late_jobs


def build_prompt(schedule: List[int], jobs: List[Dict[str, int]], fitness: int) -> str:
    """
    Build the prompt sent to the LLM at each tree search iteration.

    The prompt describes:
    - the current schedule
    - the current fitness (total weighted tardiness)
    - which jobs are late, how late, and how important

    The LLM is asked to propose 3 targeted swaps: [job_id, new_position].
    Asking for swaps (not full permutations) keeps the output small and
    reliable for a 7B model.

    Args:
        schedule: current ordered list of job IDs
        jobs: list of job dicts
        fitness: current total weighted tardiness

    Returns:
        prompt string
    """
    late_jobs = summarize_late_jobs(schedule, jobs)
    late_str = ", ".join([
        f"Job {r['job_id']} (weight={r['weight']}, tardiness={r['tardiness']}, pos={r['position']})"
        for r in late_jobs
    ]) if late_jobs else "none"

    return f"""You are solving a single machine job scheduling problem.

CURRENT SCHEDULE: {json.dumps(schedule)}
CURRENT FITNESS (Total Weighted Tardiness): {fitness} — LOWER IS BETTER
LATE JOBS: {late_str}

Propose exactly 3 swaps to improve the schedule.
Each swap moves one late job to an earlier position.
Prioritize high-weight, high-tardiness jobs.

Respond ONLY with a JSON array of 3 swaps [job_id, new_position]:
[[job_id, new_pos], [job_id, new_pos], [job_id, new_pos]]

Example: [[28, 2], [17, 5], [31, 8]]""".strip()


def _apply_swap(schedule: List[int], job_id: int, new_pos: int) -> List[int]:
    """Remove job_id from its current position and insert at new_pos."""
    s = schedule.copy()
    s.remove(job_id)
    s.insert(new_pos, job_id)
    return s


def _is_valid_schedule(schedule: List[int], n: int) -> bool:
    """Check that schedule is a valid permutation of 0..n-1."""
    return (
        isinstance(schedule, list)
        and len(schedule) == n
        and set(schedule) == set(range(n))
    )


def _fallback_schedules(schedule: List[int], n: int, k: int = 3) -> List[List[int]]:
    """
    Generate k random swap mutations as fallback when LLM output is invalid.
    Ensures the search never stops even if every LLM call fails.
    """
    results, seen = [], {tuple(schedule)}
    rng = random.Random(42)
    attempts = 0
    while len(results) < k and attempts < 200:
        s = schedule.copy()
        i, j = rng.sample(range(n), 2)
        s[i], s[j] = s[j], s[i]
        t = tuple(s)
        if t not in seen:
            results.append(s)
            seen.add(t)
        attempts += 1
    return results


def parse_and_apply(response: str, schedule: List[int], jobs: List[Dict[str, int]]) -> List[List[int]]:
    """
    Parse LLM response, extract swap proposals, apply them to the schedule.

    Handles:
    - <think>...</think> reasoning blocks (DeepSeek-R1 style)
    - ```json ... ``` markdown fences
    - Falls back to random mutations on any parse error

    Args:
        response: raw string from LLM
        schedule: current schedule to apply swaps to
        jobs: list of job dicts

    Returns:
        list of up to 3 valid child schedules
    """
    n = len(jobs)
    valid = []

    try:
        # Strategy: find the JSON array directly before any stripping
        # This works whether JSON is inside <think> tags, after them, or fenced
        json_match = re.search(r'\[\s*\[.*?\]\s*\]', response, flags=re.DOTALL)
        if json_match:
            clean = json_match.group(0)
        else:
            # Fallback: strip think tags then search again
            clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            clean = re.sub(r'```json|```', '', clean).strip()
            json_match2 = re.search(r'\[\s*\[.*?\]\s*\]', clean, flags=re.DOTALL)
            clean = json_match2.group(0) if json_match2 else clean

        swaps = json.loads(clean)

        for swap in swaps:
            try:
                job_id, new_pos = int(swap[0]), int(swap[1])
                if job_id not in schedule:
                    continue
                if not (0 <= new_pos < n):
                    continue
                s = _apply_swap(schedule, job_id, new_pos)
                if _is_valid_schedule(s, n):
                    valid.append(s)
            except Exception:
                continue

    except Exception:
        pass

    if not valid:
        valid = _fallback_schedules(schedule, n, k=3)

    return valid[:3]


def propose_schedules(schedule: List[int], jobs: List[Dict[str, int]], model) -> List[List[int]]:
    """
    Ask the LLM to propose improved schedules from the current node.

    1. Build prompt describing current state
    2. Invoke LLM
    3. Parse response and apply swaps
    4. Return up to 3 valid child schedules

    Args:
        schedule: current schedule
        jobs: list of job dicts
        model: LangChain OllamaLLM instance

    Returns:
        list of up to 3 valid child schedules
    """
    fitness = evaluate(schedule, jobs)
    prompt = build_prompt(schedule, jobs, fitness)

    try:
        response = model.invoke(prompt)
    except Exception as e:
        print(f"    LLM call failed: {e} — using fallback mutations")
        response = ""

    return parse_and_apply(response, schedule, jobs)
