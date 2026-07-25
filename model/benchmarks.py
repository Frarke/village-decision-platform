"""连续NSGA-II标准函数验证：ZDT1、ZDT2、ZDT3与DTLZ2。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    function_name: str
    seed: int
    solutions: np.ndarray
    objectives: np.ndarray
    theoretical_front: np.ndarray
    history: pd.DataFrame
    metrics: dict[str, float]


def evaluate_benchmark(name: str, x: np.ndarray) -> np.ndarray:
    name = name.upper()
    values = np.atleast_2d(np.asarray(x, dtype=float))
    if name in {"ZDT1", "ZDT2", "ZDT3"}:
        f1 = values[:, 0]
        g = 1.0 + 9.0 * values[:, 1:].mean(axis=1)
        ratio = f1 / g
        if name == "ZDT1":
            f2 = g * (1.0 - np.sqrt(ratio))
        elif name == "ZDT2":
            f2 = g * (1.0 - ratio**2)
        else:
            f2 = g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1))
        return np.column_stack([f1, f2])
    if name == "DTLZ2":
        g = np.sum((values[:, 2:] - 0.5) ** 2, axis=1)
        f1 = (1.0 + g) * np.cos(values[:, 0] * np.pi / 2) * np.cos(values[:, 1] * np.pi / 2)
        f2 = (1.0 + g) * np.cos(values[:, 0] * np.pi / 2) * np.sin(values[:, 1] * np.pi / 2)
        f3 = (1.0 + g) * np.sin(values[:, 0] * np.pi / 2)
        return np.column_stack([f1, f2, f3])
    raise ValueError(f"不支持的标准测试函数：{name}")


def theoretical_front(name: str, samples: int = 400) -> np.ndarray:
    name = name.upper()
    if name == "ZDT1":
        x = np.linspace(0.0, 1.0, samples)
        return np.column_stack([x, 1.0 - np.sqrt(x)])
    if name == "ZDT2":
        x = np.linspace(0.0, 1.0, samples)
        return np.column_stack([x, 1.0 - x**2])
    if name == "ZDT3":
        intervals = ((0.0, 0.0830015), (0.1822287, 0.2577624), (0.4093137, 0.4538821),
                     (0.6183968, 0.6525117), (0.8233318, 0.8518329))
        per = max(20, samples // len(intervals))
        x = np.concatenate([np.linspace(a, b, per) for a, b in intervals])
        return np.column_stack([x, 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)])
    if name == "DTLZ2":
        side = max(8, int(np.sqrt(samples)))
        angles = np.linspace(0.0, np.pi / 2, side)
        return np.asarray([
            [np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)]
            for a, b in product(angles, angles)
        ])
    raise ValueError(f"不支持的标准测试函数：{name}")


def _dominates_min(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    mask = np.ones(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        if not mask[i]:
            continue
        for j in range(len(objectives)):
            if i != j and _dominates_min(objectives[j], objectives[i]):
                mask[i] = False
                break
    return mask


def _sort_and_crowd(objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(objectives)
    dominated = [[] for _ in range(n)]
    counts = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    rank = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates_min(objectives[i], objectives[j]):
                dominated[i].append(j); counts[j] += 1
            elif _dominates_min(objectives[j], objectives[i]):
                dominated[j].append(i); counts[i] += 1
        if counts[i] == 0:
            fronts[0].append(i)
    level = 0
    while level < len(fronts) and fronts[level]:
        following: list[int] = []
        for i in fronts[level]:
            rank[i] = level
            for j in dominated[i]:
                counts[j] -= 1
                if counts[j] == 0:
                    following.append(j)
        if following:
            fronts.append(following)
        level += 1
    crowd = np.zeros(n)
    for front in fronts:
        if not front:
            continue
        if len(front) <= 2:
            crowd[front] = np.inf
            continue
        matrix = objectives[front]
        for column in range(matrix.shape[1]):
            order = np.argsort(matrix[:, column])
            crowd[front[order[0]]] = crowd[front[order[-1]]] = np.inf
            span = matrix[order[-1], column] - matrix[order[0], column]
            if span <= 1e-12:
                continue
            for pos in range(1, len(order) - 1):
                idx = front[order[pos]]
                if np.isfinite(crowd[idx]):
                    crowd[idx] += (matrix[order[pos + 1], column] - matrix[order[pos - 1], column]) / span
    return rank, crowd


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.ptp(b, axis=0), 1e-9)
    return np.linalg.norm((a[:, None, :] - b[None, :, :]) / scale, axis=2)


def quality_metrics(approx: np.ndarray, truth: np.ndarray, seed: int = 0) -> dict[str, float]:
    approx = np.asarray(approx, float)
    truth = np.asarray(truth, float)
    d = _distances(approx, truth)
    gd = float(np.sqrt(np.mean(np.min(d, axis=1) ** 2)))
    igd = float(np.mean(np.min(d, axis=0)))
    if len(approx) > 1:
        pair = _distances(approx, approx)
        np.fill_diagonal(pair, np.inf)
        nearest = pair.min(axis=1)
        spacing = float(np.std(nearest, ddof=1))
    else:
        spacing = float("inf")
    lower = np.min(truth, axis=0)
    span = np.maximum(np.ptp(truth, axis=0), 1e-9)
    normalized = (approx - lower) / span
    reference = np.full(approx.shape[1], 1.1)
    if approx.shape[1] == 2:
        pts = normalized[nondominated_mask(normalized)]
        pts = pts[np.argsort(pts[:, 0])]
        hv = 0.0; boundary = reference[1]
        for x, y in pts:
            if y < boundary:
                hv += max(0.0, reference[0] - x) * max(0.0, boundary - y)
                boundary = y
    else:
        rng = np.random.default_rng(seed)
        probes = rng.uniform(0.0, reference, size=(12000, len(reference)))
        dominated = np.zeros(len(probes), dtype=bool)
        for point in normalized[nondominated_mask(normalized)]:
            dominated |= np.all(point <= probes, axis=1)
        hv = float(dominated.mean() * np.prod(reference))
    return {"IGD": igd, "GD": gd, "Spacing": spacing, "Hypervolume": float(hv)}


def run_benchmark(
    name: str,
    seed: int = 42,
    population_size: int = 80,
    generations: int = 80,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> BenchmarkResult:
    name = name.upper()
    variables = 12 if name == "DTLZ2" else 30
    rng = np.random.default_rng(seed)
    population = rng.random((population_size, variables))
    objectives = evaluate_benchmark(name, population)
    history: list[dict[str, float]] = []
    truth = theoretical_front(name)

    def sbx(parent_a: np.ndarray, parent_b: np.ndarray, eta: float = 15.0) -> np.ndarray:
        child = parent_a.copy()
        for i, (left, right) in enumerate(zip(parent_a, parent_b, strict=True)):
            if rng.random() > 0.5 or abs(left - right) <= 1e-14:
                continue
            y1, y2 = min(left, right), max(left, right)
            rand = rng.random()
            beta = 1.0 + 2.0 * y1 / (y2 - y1)
            alpha = 2.0 - beta ** (-(eta + 1.0))
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            beta = 1.0 + 2.0 * (1.0 - y2) / (y2 - y1)
            alpha = 2.0 - beta ** (-(eta + 1.0))
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            child[i] = c1 if rng.random() < 0.5 else c2
        return np.clip(child, 0.0, 1.0)

    def polynomial_mutation(child: np.ndarray, eta: float = 20.0) -> np.ndarray:
        result = child.copy()
        for i, value in enumerate(result):
            if rng.random() >= 1.0 / variables:
                continue
            delta1, delta2 = value, 1.0 - value
            rand = rng.random(); power = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
                deltaq = val ** power - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
                deltaq = 1.0 - val ** power
            result[i] = np.clip(value + deltaq, 0.0, 1.0)
        return result
    for generation in range(generations + 1):
        rank, crowd = _sort_and_crowd(objectives)
        front = objectives[rank == 0]
        metrics = quality_metrics(front, truth, seed)
        item = {"迭代代数": generation, "IGD": metrics["IGD"], "非支配解数量": int(len(front))}
        history.append(item)
        if progress_callback:
            progress_callback(item)
        if generation == generations:
            break
        children: list[np.ndarray] = []
        while len(children) < population_size:
            picks = rng.integers(0, population_size, size=4)
            def choose(a: int, b: int) -> int:
                return a if (rank[a], -crowd[a]) <= (rank[b], -crowd[b]) else b
            a, b = population[choose(picks[0], picks[1])], population[choose(picks[2], picks[3])]
            children.append(polynomial_mutation(sbx(a, b)))
        combined = np.vstack([population, np.asarray(children)])
        combined_objectives = evaluate_benchmark(name, combined)
        c_rank, c_crowd = _sort_and_crowd(combined_objectives)
        order = sorted(range(len(combined)), key=lambda i: (c_rank[i], -c_crowd[i]))[:population_size]
        population = combined[order]
        objectives = combined_objectives[order]
    mask = nondominated_mask(objectives)
    final_objectives = objectives[mask]
    metrics = quality_metrics(final_objectives, truth, seed)
    return BenchmarkResult(name, seed, population[mask], final_objectives, truth, pd.DataFrame(history), metrics)


def validate_standard_functions(
    functions: tuple[str, ...] = ("ZDT1", "ZDT2", "ZDT3", "DTLZ2"),
    seeds: tuple[int, ...] = (42, 123, 2026),
    population_size: int = 80,
    generations: int = 80,
) -> tuple[pd.DataFrame, dict[str, BenchmarkResult]]:
    rows: list[dict[str, float | int | str]] = []
    examples: dict[str, BenchmarkResult] = {}
    for name in functions:
        for seed in seeds:
            result = run_benchmark(name, seed, population_size, generations)
            examples.setdefault(name, result)
            rows.append({"标准函数": name, "随机种子": seed, "非支配解数量": len(result.objectives), **result.metrics})
    return pd.DataFrame(rows), examples
