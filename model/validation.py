"""以精确枚举Pareto集为基准验证离散NSGA-II。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import numpy as np
import pandas as pd

from .pareto import OBJECTIVE_COLUMNS, dominates
from .schemas import SolveResult


@dataclass
class ValidationResult:
    metrics: pd.DataFrame
    seed_summary: pd.DataFrame
    comparison_points: pd.DataFrame
    conclusion: str


def _objective_keys(frame: pd.DataFrame, decimals: int = 6) -> set[tuple[float, ...]]:
    return {tuple(np.round(row, decimals)) for row in frame[OBJECTIVE_COLUMNS].to_numpy(float)}


def compare_exact_nsga(exact: SolveResult, nsga: SolveResult, distance_tolerance: float = 0.02) -> dict[str, float | int]:
    exact_signatures = set(exact.pareto["configuration_summary"].astype(str))
    nsga_signatures = set(nsga.pareto["configuration_summary"].astype(str))
    exact_objectives = exact.pareto[OBJECTIVE_COLUMNS].to_numpy(float)
    nsga_objectives = nsga.pareto[OBJECTIVE_COLUMNS].to_numpy(float)
    exact_ranges = np.maximum(exact_objectives.max(axis=0) - exact_objectives.min(axis=0), 1.0)
    distances = np.linalg.norm((exact_objectives[:, None, :] - nsga_objectives[None, :, :]) / exact_ranges, axis=2)
    nearest = distances.min(axis=1) if len(nsga_objectives) else np.full(len(exact_objectives), np.inf)
    dominated_count = sum(
        any(dominates(exact_point, nsga_point) for exact_point in exact_objectives)
        for nsga_point in nsga_objectives
    )
    exact_span = np.maximum(exact_objectives.max(axis=0) - exact_objectives.min(axis=0), 1e-9)
    nsga_span = nsga_objectives.max(axis=0) - nsga_objectives.min(axis=0)
    range_coverage = float(np.mean(np.minimum(1.0, nsga_span / exact_span)))
    covered = len(exact_signatures & nsga_signatures)
    return {
        "可行解比例": float(nsga.run_metadata.get("feasible_ratio", 0.0)),
        "NSGA-II非支配解数量": len(nsga.pareto),
        "精确Pareto集完全方案覆盖率": covered / max(len(exact_signatures), 1),
        "目标空间近似覆盖率": float(np.mean(nearest <= distance_tolerance)),
        "被精确Pareto集支配比例": dominated_count / max(len(nsga_objectives), 1),
        "漏解数量": len(exact_signatures - nsga_signatures),
        "目标空间范围覆盖程度": range_coverage,
        "精确枚举运行时间（秒）": float(exact.run_metadata["runtime_seconds"]),
        "NSGA-II运行时间（秒）": float(nsga.run_metadata["runtime_seconds"]),
        "末代归一化超体积": float(nsga.iteration_history.iloc[-1]["archive_hypervolume"]),
        "唯一方案评价数量": int(nsga.run_metadata.get("unique_evaluations", 0)),
        "实际搜索空间占比": float(nsga.run_metadata.get("unique_evaluations", 0)) / max(float(nsga.run_metadata.get("estimated_combinations", 1)), 1.0),
    }


def validate_runs(exact: SolveResult, nsga_runs: list[SolveResult], distance_tolerance: float = 0.02) -> ValidationResult:
    if not nsga_runs:
        raise ValueError("至少需要一次NSGA-II运行结果。")
    rows = []
    for run in nsga_runs:
        row = {"随机种子": run.run_metadata.get("random_seed"), **compare_exact_nsga(exact, run, distance_tolerance)}
        rows.append(row)
    seed_summary = pd.DataFrame(rows)
    first_metrics = rows[0]
    metrics = pd.DataFrame([
        {"验证指标": key, "本次结果": value, "定义说明": description}
        for key, value, description in [
            ("可行解比例", first_metrics["可行解比例"], "累计唯一评价方案中满足全部硬约束的比例"),
            ("NSGA-II非支配解数量", first_metrics["NSGA-II非支配解数量"], "智能搜索获得的四目标非支配方案数"),
            ("精确Pareto集完全方案覆盖率", first_metrics["精确Pareto集完全方案覆盖率"], "NSGA-II找回的精确Pareto配置数/精确Pareto配置总数"),
            ("目标空间近似覆盖率", first_metrics["目标空间近似覆盖率"], f"归一化四目标距离不超过{distance_tolerance:.3f}的精确Pareto点比例"),
            ("被精确Pareto集支配比例", first_metrics["被精确Pareto集支配比例"], "NSGA-II非支配结果中仍被精确Pareto方案严格支配的比例"),
            ("漏解数量", first_metrics["漏解数量"], "精确Pareto配置中未被NSGA-II完全找回的数量"),
            ("目标空间范围覆盖程度", first_metrics["目标空间范围覆盖程度"], "四个目标取值跨度覆盖比例的平均值"),
            ("运行时间对比", first_metrics["NSGA-II运行时间（秒）"] / max(float(first_metrics["精确枚举运行时间（秒）"]), 1e-9), "NSGA-II用时/精确枚举用时"),
            ("末代归一化超体积", first_metrics["末代归一化超体积"], "累计非支配档案在四目标0—100空间中的蒙特卡洛超体积估计，随覆盖改善总体不下降"),
            ("唯一方案评价数量", first_metrics["唯一方案评价数量"], "NSGA-II实际调用工程评价函数的不同村域方案数量"),
            ("实际搜索空间占比", first_metrics["实际搜索空间占比"], "唯一方案评价数量/理论组合总数"),
        ]
    ])
    objective_sets = [_objective_keys(run.pareto) for run in nsga_runs]
    jaccards = []
    for index in range(len(objective_sets)):
        for other in range(index + 1, len(objective_sets)):
            union = objective_sets[index] | objective_sets[other]
            jaccards.append(len(objective_sets[index] & objective_sets[other]) / max(len(union), 1))
    stability = fmean(jaccards) if jaccards else 1.0
    seed_summary["跨种子目标集稳定性"] = stability
    first = nsga_runs[0].pareto.copy()
    first["解集来源"] = "NSGA-II智能搜索"
    exact_points = exact.pareto.copy()
    exact_points["解集来源"] = "精确枚举基准"
    comparison = pd.concat([exact_points, first], ignore_index=True, sort=False)
    coverage = float(first_metrics["目标空间近似覆盖率"])
    dominated = float(first_metrics["被精确Pareto集支配比例"])
    conclusion = (
        f"在当前参数下，NSGA-II目标空间近似覆盖率为{coverage:.1%}，"
        f"被精确Pareto集支配的比例为{dominated:.1%}，跨随机种子目标集稳定性为{stability:.1%}。"
        "该结论仅评价算法在当前示例问题上的搜索表现，不替代工程参数标定与方案专家复核。"
    )
    return ValidationResult(metrics, seed_summary, comparison, conclusion)
