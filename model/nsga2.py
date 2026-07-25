"""与工程评价函数直接耦合的离散NSGA-II求解器。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd

from .combinations import estimate_enumerations, generate_unit_options
from .constraints import check_plan_constraints
from .evaluation import evaluate_unit
from .pareto import OBJECTIVE_COLUMNS, dominates, pareto_front
from .recommendation import recommend
from .regional import evaluate_region
from .schemas import GovernanceUnit, Measure, SolveResult, SolverConfig, validate_measures, validate_units
from .screening import screen_all
from .solver import run_identity
from .tagging import in_governance_library, tagged_unit_record


def _violation_score(reasons: list[str], total_cost: float, config: SolverConfig) -> float:
    score = max(0.0, total_cost - config.total_budget) / max(config.total_budget, 1.0)
    for reason in reasons:
        if "预算" in reason:
            continue
        if "未治理" in reason:
            score += 1.0
        elif "FS" in reason:
            score += 0.7
        elif "安全得分" in reason:
            score += 0.7
        else:
            score += 0.5
    return score


def _constraint_dominates(a: dict[str, object], b: dict[str, object]) -> bool:
    if bool(a["feasible"]) != bool(b["feasible"]):
        return bool(a["feasible"])
    if not bool(a["feasible"]):
        return float(a["violation"]) < float(b["violation"]) - 1e-12
    return dominates(np.asarray(a["objectives"]), np.asarray(b["objectives"]))


def _fast_sort(population: list[dict[str, object]]) -> list[list[int]]:
    dominated_sets = [[] for _ in population]
    domination_counts = [0 for _ in population]
    fronts: list[list[int]] = [[]]
    for i, left in enumerate(population):
        for j, right in enumerate(population):
            if i == j:
                continue
            if _constraint_dominates(left, right):
                dominated_sets[i].append(j)
            elif _constraint_dominates(right, left):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            left["rank"] = 0
            fronts[0].append(i)
    level = 0
    while level < len(fronts) and fronts[level]:
        next_front: list[int] = []
        for i in fronts[level]:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    population[j]["rank"] = level + 1
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        level += 1
    return fronts


def _crowding(population: list[dict[str, object]], front: list[int]) -> None:
    for index in front:
        population[index]["crowding"] = 0.0
    if len(front) <= 2:
        for index in front:
            population[index]["crowding"] = float("inf")
        return
    if all(bool(population[index]["feasible"]) for index in front):
        matrix = np.asarray([population[index]["objectives"] for index in front], dtype=float)
    else:
        matrix = np.asarray([[-float(population[index]["violation"])] for index in front])
    for column in range(matrix.shape[1]):
        order = np.argsort(matrix[:, column])
        population[front[int(order[0])]]["crowding"] = float("inf")
        population[front[int(order[-1])]]["crowding"] = float("inf")
        span = matrix[int(order[-1]), column] - matrix[int(order[0]), column]
        if span <= 1e-12:
            continue
        for position in range(1, len(order) - 1):
            index = front[int(order[position])]
            if np.isinf(float(population[index]["crowding"])):
                continue
            distance = (matrix[int(order[position + 1]), column] - matrix[int(order[position - 1]), column]) / span
            population[index]["crowding"] = float(population[index]["crowding"]) + float(distance)


def _rank_and_crowd(population: list[dict[str, object]]) -> list[list[int]]:
    fronts = _fast_sort(population)
    for front in fronts:
        _crowding(population, front)
    return fronts


def _tournament(population: list[dict[str, object]], rng: np.random.Generator) -> dict[str, object]:
    left, right = rng.integers(0, len(population), size=2)
    a, b = population[int(left)], population[int(right)]
    key_a = (int(a["rank"]), -float(a["crowding"]))
    key_b = (int(b["rank"]), -float(b["crowding"]))
    return a if key_a <= key_b else b


def solve_nsga2(
    units: list[GovernanceUnit],
    measures: list[Measure],
    config: SolverConfig | None = None,
    assumptions_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> SolveResult:
    started = time.perf_counter()
    config = config or SolverConfig()
    errors = validate_units(units) + validate_measures(measures)
    if errors:
        raise ValueError("\n".join(errors))
    if config.nsga_population_size < 20 or config.nsga_generations < 1:
        raise ValueError("NSGA-II种群规模至少为20，最大迭代代数至少为1。")
    rng = np.random.default_rng(config.random_seed)
    active_units = [unit for unit in units if in_governance_library(unit)[0]]
    screening_results = screen_all(active_units, measures)
    screening_frame = pd.DataFrame([item.to_dict() for item in screening_results])
    tagged_frame = pd.DataFrame([tagged_unit_record(unit) for unit in units])
    options_by_unit = {
        unit.unit_id: generate_unit_options(unit, measures, screening_results, config) for unit in active_units
    }
    if any(not values for values in options_by_unit.values()):
        raise ValueError("至少一个治理单元没有满足准入与组合约束的候选配置。")
    option_lists = [options_by_unit[unit.unit_id] for unit in active_units]
    bounds = [len(values) for values in option_lists]
    search_space = estimate_enumerations(options_by_unit)
    cache: dict[tuple[int, ...], dict[str, object]] = {}

    def evaluate(genes: tuple[int, ...]) -> dict[str, object]:
        if genes in cache:
            return cache[genes]
        selected_by_unit = [option_lists[i][gene] for i, gene in enumerate(genes)]
        details = [evaluate_unit(unit, tuple(selected), config) for unit, selected in zip(active_units, selected_by_unit, strict=True)]
        region = evaluate_region(active_units, details)
        feasible, reasons = check_plan_constraints(active_units, details, region["total_cost"], config)
        configuration = "；".join(f"{item['unit_name']}={item['measures']}" for item in details)
        review = sorted({str(item["expert_review"]) for item in details if item["expert_review"] != "无"})
        row: dict[str, object] = {
            **region, "configuration_summary": configuration,
            "expert_review": "；".join(review) if review else "无", "_genes": genes,
        }
        for item in details:
            row[f"unit_{item['unit_id']}"] = item["measures"]
        record = {
            "genes": genes, "details": details, "row": row, "feasible": feasible,
            "reasons": reasons, "violation": _violation_score(reasons, float(region["total_cost"]), config),
            "objectives": np.asarray([region[name] for name in OBJECTIVE_COLUMNS], dtype=float),
        }
        cache[genes] = record
        return record

    def random_genes() -> tuple[int, ...]:
        return tuple(int(rng.integers(0, upper)) for upper in bounds)

    def repair(genes: tuple[int, ...]) -> tuple[int, ...]:
        """以少量局部候选降低约束违反度，避免一次修复遍历完整邻域。"""
        current = evaluate(genes)
        for _ in range(max(2, len(bounds))):
            if bool(current["feasible"]):
                return tuple(current["genes"])
            neighbors = []
            base = list(current["genes"])
            for _candidate in range(min(4, len(bounds) + 1)):
                position = int(rng.integers(0, len(bounds)))
                upper = bounds[position]
                candidate = base.copy()
                candidate[position] = int(rng.integers(0, upper))
                if candidate[position] != base[position]:
                    neighbors.append(evaluate(tuple(candidate)))
            if not neighbors:
                break
            best = min(neighbors, key=lambda item: (float(item["violation"]), -float(np.sum(item["objectives"]))))
            if float(best["violation"]) >= float(current["violation"]) - 1e-12:
                break
            current = best
        return tuple(current["genes"])

    # 初始种群同时包含低索引基线、高索引强措施基线与随机个体。
    seed_genes = [tuple(0 for _ in bounds), tuple(upper - 1 for upper in bounds)]
    while len(seed_genes) < config.nsga_population_size:
        seed_genes.append(random_genes())
    population = [evaluate(repair(genes)).copy() for genes in seed_genes[:config.nsga_population_size]]
    history: list[dict[str, object]] = []
    hv_probes = np.random.default_rng(20260722).uniform(0.0, 100.0, size=(3000, len(OBJECTIVE_COLUMNS)))
    archive_front = np.empty((0, len(OBJECTIVE_COLUMNS)), dtype=float)

    for generation in range(config.nsga_generations + 1):
        fronts = _rank_and_crowd(population)
        feasible_population = [item for item in population if bool(item["feasible"])]
        first_feasible = [population[index] for index in fronts[0] if bool(population[index]["feasible"])] if fronts else []
        population_matrix = np.asarray([item["objectives"] for item in feasible_population], dtype=float)
        candidates = np.vstack([archive_front, population_matrix]) if len(population_matrix) else archive_front
        if len(candidates):
            survivors: list[np.ndarray] = []
            for point in candidates:
                if any(np.array_equal(point, prior) or dominates(prior, point) for prior in survivors):
                    continue
                survivors = [prior for prior in survivors if not dominates(point, prior)]
                survivors.append(point)
            archive_front = np.asarray(survivors, dtype=float)
            dominated_probe = np.zeros(len(hv_probes), dtype=bool)
            for point in archive_front:
                dominated_probe |= np.all(point >= hv_probes, axis=1)
            hypervolume = float(dominated_probe.mean())
            if len(archive_front) > 1:
                scale = np.maximum(np.ptp(archive_front, axis=0), 1.0)
                distance = np.linalg.norm((archive_front[:, None] - archive_front[None, :]) / scale, axis=2)
                np.fill_diagonal(distance, np.inf)
                spacing = float(np.std(distance.min(axis=1)))
            else:
                spacing = 0.0
        else:
            hypervolume, spacing = 0.0, 0.0
        history.append({
            "generation": generation, "current_feasible": len(feasible_population),
            "current_nondominated": len(first_feasible), "unique_evaluations": len(cache),
            "archive_hypervolume": round(hypervolume, 6), "front_spacing": round(spacing, 6),
            "search_space_ratio": round(len(cache) / max(search_space, 1), 6),
        })
        if progress_callback is not None:
            progress_callback(history[-1])
        if generation == config.nsga_generations:
            break
        offspring: list[dict[str, object]] = []
        while len(offspring) < config.nsga_population_size:
            parent_a = tuple(_tournament(population, rng)["genes"])
            parent_b = tuple(_tournament(population, rng)["genes"])
            child_a, child_b = list(parent_a), list(parent_b)
            if rng.random() < config.nsga_crossover_probability:
                for position in range(len(bounds)):
                    if rng.random() < 0.5:
                        child_a[position], child_b[position] = child_b[position], child_a[position]
            for child in (child_a, child_b):
                for position, upper in enumerate(bounds):
                    if rng.random() < config.nsga_mutation_probability:
                        child[position] = int(rng.integers(0, upper))
                genes = repair(tuple(child))
                offspring.append(evaluate(genes).copy())
                if len(offspring) >= config.nsga_population_size:
                    break
        # 每代引入少量随机移民，降低离散小空间的早熟收敛风险。
        immigrants = max(2, config.nsga_population_size // 20)
        offspring[-immigrants:] = [evaluate(repair(random_genes())).copy() for _ in range(immigrants)]
        combined = population + offspring
        combined_fronts = _rank_and_crowd(combined)
        next_population: list[dict[str, object]] = []
        for front in combined_fronts:
            if len(next_population) + len(front) <= config.nsga_population_size:
                next_population.extend(combined[index] for index in front)
            else:
                ordered = sorted(front, key=lambda index: float(combined[index]["crowding"]), reverse=True)
                remaining = config.nsga_population_size - len(next_population)
                next_population.extend(combined[index] for index in ordered[:remaining])
                break
        population = next_population

    feasible_records = [item for item in cache.values() if bool(item["feasible"])]
    if not feasible_records:
        raise ValueError("NSGA-II未搜索到满足全部硬约束的方案，请增加种群/代数或复核预算与安全底线。")
    feasible_records.sort(key=lambda item: tuple(item["genes"]))
    rows = []
    detail_lookup: dict[str, list[dict[str, object]]] = {}
    for index, record in enumerate(feasible_records, start=1):
        plan_id = f"N{index:05d}"
        row = {"plan_id": plan_id, **{k: v for k, v in record["row"].items() if k != "_genes"}}
        rows.append(row)
        detail_lookup[plan_id] = record["details"]
    feasible = pd.DataFrame(rows)
    pareto = pareto_front(feasible)
    representatives = recommend(pareto, config.preference_weights)
    detail_rows = []
    for _, recommendation in representatives.iterrows():
        for detail in detail_lookup[str(recommendation["plan_id"])]:
            detail_rows.append({"recommendation_type": recommendation["recommendation_type"], "plan_id": recommendation["plan_id"], **detail})
    assumptions = pd.read_csv(assumptions_path) if assumptions_path and Path(assumptions_path).exists() else pd.DataFrame()
    reason_counts: dict[str, int] = {}
    for item in cache.values():
        for reason in item["reasons"]:
            category = "预算约束" if "预算" in reason else "安全底线或治理约束"
            reason_counts[category] = reason_counts.get(category, 0) + 1
    runtime = time.perf_counter() - started
    metadata = {
        **run_identity(units, measures, config, "nsga2_discrete"), "solver": "nsga2_discrete",
        "deterministic": True, "active_units": len(active_units), "estimated_combinations": search_space,
        "unique_evaluations": len(cache), "feasible_evaluations": len(feasible_records),
        "feasible_ratio": round(len(feasible_records) / max(len(cache), 1), 6),
        "feasible_plans": len(feasible), "pareto_plans": len(pareto), "runtime_seconds": round(runtime, 6),
        "config": asdict(config),
    }
    return SolveResult(
        tagged_units=tagged_frame, measure_library=pd.DataFrame([measure.to_dict() for measure in measures]),
        screening=screening_frame, all_feasible=feasible, pareto=pareto,
        representatives=representatives, representative_details=pd.DataFrame(detail_rows),
        run_metadata=metadata, assumptions=assumptions, iteration_history=pd.DataFrame(history),
        constraint_statistics=pd.DataFrame([{"约束类型": key, "触发次数": value} for key, value in sorted(reason_counts.items())]),
        detail_lookup=detail_lookup,
    )
