# src/search.py
from typing import Dict, List, Tuple

from src.baselines import edf_schedule, spt_schedule, wspt_schedule
from src.fitness import evaluate
from src.llm import propose_schedules


def tree_of_thoughts_search(
    jobs: List[Dict[str, int]],
    model,
    max_depth: int = 5,
    beam_width: int = 3,
    proposals_per_node: int = 3,
    verbose: bool = True,
) -> Tuple[List[int], int, List[Dict]]:
    """
    LLM-guided Tree of Thoughts beam search for single machine scheduling.

    Algorithm:
    1. Seed the beam with the best classical baseline (EDF, SPT, or WSPT)
    2. At each depth:
       a. For each node in the beam, ask the LLM to propose k child schedules
       b. Evaluate all children with the exact weighted tardiness objective
       c. Keep the top beam_width children as the new beam
       d. Track the global best schedule seen so far
    3. Stop at max_depth or when no new children can be generated

    The LLM proposes targeted swaps [job_id, new_position] rather than
    full permutations — this is reliable for 7B models and keeps output small.
    Python applies the swaps and evaluates with the exact fitness function.

    Args:
        jobs: list of job dicts (id, processing_time, deadline, weight)
        model: LangChain OllamaLLM instance (deepseek-r1:7b recommended)
        max_depth: maximum number of search iterations
        beam_width: number of top candidates to keep at each depth
        proposals_per_node: number of child schedules to generate per node
        verbose: print progress at each depth

    Returns:
        best_schedule: list of job IDs in optimal order found
        best_fitness: total weighted tardiness of best schedule
        history: list of dicts with depth, best_fitness, beam_size per iteration
    """
    # ── seed: pick best classical baseline ──────────────────────────────────
    seed_candidates = {
        "EDF": edf_schedule(jobs),
        "SPT": spt_schedule(jobs),
        "WSPT": wspt_schedule(jobs),
    }
    best_seed_name = min(seed_candidates, key=lambda name: evaluate(seed_candidates[name], jobs))
    best_seed_schedule = seed_candidates[best_seed_name]

    beam = [{
        "schedule": best_seed_schedule,
        "fitness": evaluate(best_seed_schedule, jobs),
    }]
    best_schedule = beam[0]["schedule"]
    best_fitness = beam[0]["fitness"]
    seen = {tuple(best_schedule)}

    history = [{"depth": 0, "best_fitness": best_fitness, "beam_size": 1}]

    if verbose:
        print(f"Depth 0 | Best fitness: {best_fitness} | Beam size: 1 | Seed: {best_seed_name}")

    # ── search loop ──────────────────────────────────────────────────────────
    for depth in range(1, max_depth + 1):
        all_children = []

        for node_idx, node in enumerate(beam):
            proposals = propose_schedules(node["schedule"], jobs, model)

            for child_schedule in proposals:
                child_tuple = tuple(child_schedule)
                if child_tuple not in seen:
                    all_children.append({
                        "schedule": child_schedule,
                        "fitness": evaluate(child_schedule, jobs),
                    })
                    seen.add(child_tuple)

        if not all_children:
            if verbose:
                print(f"Depth {depth} | No new children generated — stopping early.")
            break

        # rank and prune
        all_children.sort(key=lambda x: x["fitness"])
        beam = all_children[:beam_width]

        if beam[0]["fitness"] < best_fitness:
            best_fitness = beam[0]["fitness"]
            best_schedule = beam[0]["schedule"]

        history.append({"depth": depth, "best_fitness": best_fitness, "beam_size": len(beam)})

        if verbose:
            print(f"Depth {depth} | Best fitness: {best_fitness} | Beam size: {len(beam)}")

    return best_schedule, best_fitness, history
