"""连续NSGA-II标准测试函数验证工作台。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    objectives: tuple[int, ...]
    suggested_variables: int
    variable_range: str
    front_description: str
    capability: str


BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "ZDT1": BenchmarkSpec("ZDT1", (2,), 30, "[0, 1]", "凸Pareto前沿", "基础收敛与分布能力"),
    "ZDT2": BenchmarkSpec("ZDT2", (2,), 30, "[0, 1]", "非凸Pareto前沿", "非凸前沿逼近能力"),
    "ZDT3": BenchmarkSpec("ZDT3", (2,), 30, "[0, 1]", "不连续Pareto前沿", "多段前沿保持能力"),
    "ZDT4": BenchmarkSpec("ZDT4", (2,), 10, "x1∈[0,1], 其余∈[-5,5]", "多峰复杂前沿", "局部极值逃逸能力"),
    "ZDT6": BenchmarkSpec("ZDT6", (2,), 10, "[0, 1]", "前沿非均匀", "非均匀分布搜索能力"),
    "DTLZ1": BenchmarkSpec("DTLZ1", (3, 4), 12, "[0, 1]", "线性超平面", "高维线性前沿覆盖能力"),
    "DTLZ2": BenchmarkSpec("DTLZ2", (3, 4), 12, "[0, 1]", "球面Pareto前沿", "高维平滑前沿覆盖能力"),
    "DTLZ3": BenchmarkSpec("DTLZ3", (3, 4), 12, "[0, 1]", "多峰球面前沿", "复杂高维收敛能力"),
    "DTLZ4": BenchmarkSpec("DTLZ4", (3, 4), 12, "[0, 1]", "偏置球面前沿", "分布均匀性与偏置鲁棒性"),
    "DTLZ7": BenchmarkSpec("DTLZ7", (3, 4), 12, "[0, 1]", "离散不连续高维前沿", "不连续高维前沿搜索能力"),
}


@dataclass
class BenchmarkResult:
    function_name: str
    objectives_count: int
    variables_count: int
    seed: int
    solutions: np.ndarray
    objectives: np.ndarray
    theoretical_front: np.ndarray
    history: pd.DataFrame
    metrics: dict[str, float | None]


def benchmark_functions_for_objectives(objectives_count: int) -> list[str]:
    return [name for name, spec in BENCHMARK_SPECS.items() if objectives_count in spec.objectives]


def benchmark_spec_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "标准函数": spec.name,
                "支持目标数": "、".join(str(item) for item in spec.objectives),
                "建议决策变量维度": spec.suggested_variables,
                "变量范围": spec.variable_range,
                "Pareto前沿特征": spec.front_description,
                "适合检验的能力": spec.capability,
            }
            for spec in BENCHMARK_SPECS.values()
        ]
    )


def _dtlz_g(values: np.ndarray, offset: int) -> np.ndarray:
    return np.sum((values[:, offset:] - 0.5) ** 2, axis=1)


def evaluate_benchmark(name: str, x: np.ndarray, objectives_count: int | None = None) -> np.ndarray:
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
    if name == "ZDT4":
        f1 = values[:, 0]
        g = 1.0 + 10.0 * (values.shape[1] - 1) + np.sum(values[:, 1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * values[:, 1:]), axis=1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.column_stack([f1, f2])
    if name == "ZDT6":
        f1 = 1.0 - np.exp(-4.0 * values[:, 0]) * np.sin(6.0 * np.pi * values[:, 0]) ** 6
        g = 1.0 + 9.0 * (np.mean(values[:, 1:], axis=1) ** 0.25)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.column_stack([f1, f2])

    objectives = int(objectives_count or 3)
    if objectives < 3:
        raise ValueError(f"{name}至少需要3个目标。")
    if name in {"DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ7"}:
        k = values.shape[1] - objectives + 1
        if k <= 0:
            raise ValueError(f"{name}的决策变量维度不足以支持{objectives}个目标。")
    if name == "DTLZ1":
        g = 100.0 * (k + np.sum((values[:, objectives - 1:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (values[:, objectives - 1:] - 0.5)), axis=1))
        result = np.full((len(values), objectives), 0.5 * (1.0 + g)[:, None])
        for idx in range(objectives):
            for col in range(objectives - idx - 1):
                result[:, idx] *= values[:, col]
            if idx > 0:
                result[:, idx] *= 1.0 - values[:, objectives - idx - 1]
        return result
    if name in {"DTLZ2", "DTLZ3", "DTLZ4"}:
        if name == "DTLZ2":
            g = _dtlz_g(values, objectives - 1)
            angles = values[:, : objectives - 1] * np.pi / 2.0
        elif name == "DTLZ3":
            g = 100.0 * (k + np.sum((values[:, objectives - 1:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (values[:, objectives - 1:] - 0.5)), axis=1))
            angles = values[:, : objectives - 1] * np.pi / 2.0
        else:
            g = _dtlz_g(values, objectives - 1)
            angles = (values[:, : objectives - 1] ** 100) * np.pi / 2.0
        result = np.full((len(values), objectives), 1.0 + g[:, None])
        for idx in range(objectives):
            for col in range(objectives - idx - 1):
                result[:, idx] *= np.cos(angles[:, col])
            if idx > 0:
                result[:, idx] *= np.sin(angles[:, objectives - idx - 1])
        return result
    if name == "DTLZ7":
        g = 1.0 + 9.0 / k * np.sum(values[:, objectives - 1:], axis=1)
        result = np.zeros((len(values), objectives))
        result[:, : objectives - 1] = values[:, : objectives - 1]
        h = objectives - np.sum(
            (result[:, : objectives - 1] / (1.0 + g[:, None])) * (1.0 + np.sin(3.0 * np.pi * result[:, : objectives - 1])),
            axis=1,
        )
        result[:, objectives - 1] = (1.0 + g) * h
        return result
    raise ValueError(f"不支持的标准测试函数：{name}")


def theoretical_front(name: str, objectives_count: int, samples: int = 400) -> np.ndarray:
    name = name.upper()
    if name == "ZDT1":
        x = np.linspace(0.0, 1.0, samples)
        return np.column_stack([x, 1.0 - np.sqrt(x)])
    if name == "ZDT2":
        x = np.linspace(0.0, 1.0, samples)
        return np.column_stack([x, 1.0 - x**2])
    if name == "ZDT3":
        intervals = ((0.0, 0.0830015), (0.1822287, 0.2577624), (0.4093137, 0.4538821), (0.6183968, 0.6525117), (0.8233318, 0.8518329))
        per = max(20, samples // len(intervals))
        x = np.concatenate([np.linspace(a, b, per) for a, b in intervals])
        return np.column_stack([x, 1.0 - np.sqrt(x) - x * np.sin(10.0 * np.pi * x)])
    if name == "ZDT4":
        x = np.linspace(0.0, 1.0, samples)
        return np.column_stack([x, 1.0 - np.sqrt(x)])
    if name == "ZDT6":
        x = np.linspace(0.280775, 1.0, samples)
        f1 = 1.0 - np.exp(-4.0 * x) * np.sin(6.0 * np.pi * x) ** 6
        return np.column_stack([f1, 1.0 - f1**2])

    rng = np.random.default_rng(20260727)
    if name == "DTLZ1":
        raw = rng.dirichlet(np.ones(objectives_count), size=samples) * 0.5
        return raw
    if name in {"DTLZ2", "DTLZ3", "DTLZ4"}:
        raw = rng.random((samples, objectives_count))
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / np.maximum(norm, 1e-12)
    if name == "DTLZ7":
        raw = rng.random((samples, objectives_count))
        raw[:, -1] = objectives_count - np.sum((raw[:, :-1] / 2.0) * (1.0 + np.sin(3.0 * np.pi * raw[:, :-1])), axis=1)
        return raw
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
                dominated[i].append(j)
                counts[j] += 1
            elif _dominates_min(objectives[j], objectives[i]):
                dominated[j].append(i)
                counts[i] += 1
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


def quality_metrics(approx: np.ndarray, truth: np.ndarray, seed: int = 0) -> dict[str, float | None]:
    approx = np.asarray(approx, float)
    truth = np.asarray(truth, float)
    if len(approx) == 0 or len(truth) == 0:
        return {"IGD": None, "GD": None, "Spacing": None, "Hypervolume": None}
    d = _distances(approx, truth)
    gd = float(np.sqrt(np.mean(np.min(d, axis=1) ** 2)))
    igd = float(np.mean(np.min(d, axis=0)))
    if len(approx) > 1:
        pair = _distances(approx, approx)
        np.fill_diagonal(pair, np.inf)
        nearest = pair.min(axis=1)
        spacing = float(np.std(nearest, ddof=1))
    else:
        spacing = 0.0
    lower = np.min(truth, axis=0)
    span = np.maximum(np.ptp(truth, axis=0), 1e-9)
    normalized = (approx - lower) / span
    reference = np.full(approx.shape[1], 1.1)
    rng = np.random.default_rng(seed)
    probes = rng.uniform(0.0, reference, size=(15000, len(reference)))
    dominated = np.zeros(len(probes), dtype=bool)
    for point in normalized[nondominated_mask(normalized)]:
        dominated |= np.all(point <= probes, axis=1)
    hv = float(dominated.mean() * np.prod(reference))
    return {"IGD": igd, "GD": gd, "Spacing": spacing, "Hypervolume": hv}


def _variable_bounds(name: str, variables: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(variables)
    upper = np.ones(variables)
    if name.upper() == "ZDT4":
        lower[1:] = -5.0
        upper[1:] = 5.0
    return lower, upper


def run_benchmark(
    name: str,
    objectives_count: int,
    variables_count: int,
    seed: int = 42,
    population_size: int = 80,
    generations: int = 80,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> BenchmarkResult:
    name = name.upper()
    if name not in BENCHMARK_SPECS:
        raise ValueError(f"未知标准函数：{name}")
    spec = BENCHMARK_SPECS[name]
    if objectives_count not in spec.objectives:
        raise ValueError(f"{name}不支持{objectives_count}目标。")
    rng = np.random.default_rng(seed)
    lower, upper = _variable_bounds(name, variables_count)
    population = rng.uniform(lower, upper, size=(population_size, variables_count))
    objectives = evaluate_benchmark(name, population, objectives_count)
    history: list[dict[str, float]] = []
    truth = theoretical_front(name, objectives_count)

    def sbx(parent_a: np.ndarray, parent_b: np.ndarray, eta: float = 15.0) -> np.ndarray:
        child = parent_a.copy()
        for i, (left, right) in enumerate(zip(parent_a, parent_b, strict=True)):
            if rng.random() > 0.5 or abs(left - right) <= 1e-14:
                continue
            low_i = lower[i]
            high_i = upper[i]
            y1, y2 = min(left, right), max(left, right)
            rand = rng.random()
            beta = 1.0 + 2.0 * (y1 - low_i) / (y2 - y1)
            alpha = 2.0 - beta ** (-(eta + 1.0))
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            beta = 1.0 + 2.0 * (high_i - y2) / (y2 - y1)
            alpha = 2.0 - beta ** (-(eta + 1.0))
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            child[i] = c1 if rng.random() < 0.5 else c2
            child[i] = np.clip(child[i], low_i, high_i)
        return child

    def polynomial_mutation(child: np.ndarray, eta: float = 20.0) -> np.ndarray:
        result = child.copy()
        for i, value in enumerate(result):
            if rng.random() >= 1.0 / variables_count:
                continue
            low_i = lower[i]
            high_i = upper[i]
            span_i = high_i - low_i
            normalized = (value - low_i) / max(span_i, 1e-12)
            delta1 = normalized
            delta2 = 1.0 - normalized
            rand = rng.random()
            power = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1.0)
                deltaq = val ** power - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1.0)
                deltaq = 1.0 - val ** power
            result[i] = np.clip(value + deltaq * span_i, low_i, high_i)
        return result

    for generation in range(generations + 1):
        rank, crowd = _sort_and_crowd(objectives)
        front = objectives[rank == 0]
        metrics = quality_metrics(front, truth, seed)
        item = {
            "迭代代数": generation,
            "IGD": metrics["IGD"] if metrics["IGD"] is not None else np.nan,
            "非支配解数量": int(len(front)),
            "Hypervolume": metrics["Hypervolume"] if metrics["Hypervolume"] is not None else np.nan,
        }
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

            a = population[choose(picks[0], picks[1])]
            b = population[choose(picks[2], picks[3])]
            children.append(polynomial_mutation(sbx(a, b)))
        combined = np.vstack([population, np.asarray(children)])
        combined_objectives = evaluate_benchmark(name, combined, objectives_count)
        c_rank, c_crowd = _sort_and_crowd(combined_objectives)
        order = sorted(range(len(combined)), key=lambda i: (c_rank[i], -c_crowd[i]))[:population_size]
        population = combined[order]
        objectives = combined_objectives[order]

    mask = nondominated_mask(objectives)
    final_objectives = objectives[mask]
    metrics = quality_metrics(final_objectives, truth, seed)
    return BenchmarkResult(name, objectives_count, variables_count, seed, population[mask], final_objectives, truth, pd.DataFrame(history), metrics)


def validate_standard_functions(
    functions: tuple[str, ...] = ("ZDT1", "ZDT2", "ZDT3", "DTLZ2"),
    seeds: tuple[int, ...] = (42, 123, 2026),
    population_size: int = 80,
    generations: int = 80,
    objectives_count: int | None = None,
    variables_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, BenchmarkResult]]:
    rows: list[dict[str, float | int | str | None]] = []
    examples: dict[str, BenchmarkResult] = {}
    for name in functions:
        spec = BENCHMARK_SPECS[name.upper()]
        current_objectives = objectives_count or spec.objectives[0]
        current_variables = variables_count or spec.suggested_variables
        for seed in seeds:
            result = run_benchmark(name, current_objectives, current_variables, seed, population_size, generations)
            examples.setdefault(name.upper(), result)
            rows.append(
                {
                    "标准函数": name.upper(),
                    "目标数": current_objectives,
                    "决策变量维度": current_variables,
                    "随机种子": seed,
                    "非支配解数量": len(result.objectives),
                    **result.metrics,
                }
            )
    return pd.DataFrame(rows), examples
