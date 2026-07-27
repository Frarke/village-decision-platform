from __future__ import annotations

import numpy as np
import pandas as pd

from .pareto import OBJECTIVE_COLUMNS


CATEGORY_RULES = {
    "安全性最高代表方案": {
        "primary": [("safety", False)],
        "tiebreak": [("resilience", False), ("total_cost", True), ("economy", False), ("sustainability", False), ("plan_id", True)],
        "basis": "先比较安全性，再依次比较韧性、总成本、经济性、可持续性和方案编号。",
    },
    "经济性最高代表方案": {
        "primary": [("economy", False)],
        "tiebreak": [("total_cost", True), ("safety", False), ("resilience", False), ("sustainability", False), ("plan_id", True)],
        "basis": "先比较经济性评分SC，再依次比较总成本、安全性、韧性、可持续性和方案编号。",
    },
    "可持续性最高代表方案": {
        "primary": [("sustainability", False)],
        "tiebreak": [("safety", False), ("resilience", False), ("total_cost", True), ("economy", False), ("plan_id", True)],
        "basis": "先比较可持续性，再依次比较安全性、韧性、总成本、经济性和方案编号。",
    },
    "韧性最高代表方案": {
        "primary": [("resilience", False)],
        "tiebreak": [("safety", False), ("sustainability", False), ("total_cost", True), ("economy", False), ("plan_id", True)],
        "basis": "先比较韧性，再依次比较安全性、可持续性、总成本、经济性和方案编号。",
    },
}


def _stable_candidates(frame: pd.DataFrame, columns: list[tuple[str, bool]]) -> pd.DataFrame:
    return frame.sort_values(
        by=[name for name, _ in columns],
        ascending=[ascending for _, ascending in columns],
        kind="mergesort",
    ).reset_index(drop=True)


def _pick_by_rule(frame: pd.DataFrame, label: str) -> dict[str, object]:
    rule = CATEGORY_RULES[label]
    primary_name, primary_ascending = rule["primary"][0]
    primary_sorted = _stable_candidates(frame, rule["primary"])
    best_value = float(primary_sorted.iloc[0][primary_name])
    tied = primary_sorted[np.isclose(primary_sorted[primary_name].astype(float), best_value)].copy()
    selected = _stable_candidates(tied, rule["tiebreak"]).iloc[0]
    row = selected.to_dict()
    row["recommendation_type"] = label
    row["recommendation_basis"] = rule["basis"]
    row["tie_candidate_count"] = int(len(tied))
    row["tie_candidate_ids"] = "、".join(tied["plan_id"].astype(str).tolist())
    row["tie_break_rule"] = " > ".join(name for name, _ in rule["tiebreak"])
    row["primary_indicator"] = primary_name
    row["primary_value"] = float(best_value)
    row["primary_ascending"] = primary_ascending
    return row


def _compromise_row(frame: pd.DataFrame, weights: dict[str, float]) -> dict[str, object]:
    enriched = frame.copy()
    matrix = enriched[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    low = matrix.min(axis=0)
    span = matrix.max(axis=0) - low
    normalized = np.where(span > 0, (matrix - low) / np.where(span == 0, 1.0, span), 1.0)
    raw_weights = np.array([max(0.0, float(weights.get(name, 0.0))) for name in OBJECTIVE_COLUMNS], dtype=float)
    if raw_weights.sum() <= 0:
        raw_weights = np.ones(len(OBJECTIVE_COLUMNS), dtype=float)
    raw_weights /= raw_weights.sum()
    distances = np.sqrt(((1.0 - normalized) ** 2 * raw_weights).sum(axis=1))
    enriched["compromise_distance"] = distances
    enriched["weakest_objective"] = matrix.min(axis=1)
    sorted_frame = enriched.sort_values(
        by=["compromise_distance", "weakest_objective", "safety", "resilience", "economy", "sustainability", "total_cost", "plan_id"],
        ascending=[True, False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best_distance = float(sorted_frame.iloc[0]["compromise_distance"])
    tied = sorted_frame[np.isclose(sorted_frame["compromise_distance"].astype(float), best_distance)].copy()
    selected = tied.sort_values(
        by=["weakest_objective", "safety", "resilience", "economy", "sustainability", "total_cost", "plan_id"],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    row = selected.to_dict()
    row["recommendation_type"] = "综合折中代表方案"
    row["recommendation_basis"] = "在Pareto候选集中按四目标同向归一化后，基于偏好权重选择与理想点加权距离最小的唯一方案。"
    row["tie_candidate_count"] = int(len(tied))
    row["tie_candidate_ids"] = "、".join(tied["plan_id"].astype(str).tolist())
    row["tie_break_rule"] = "weakest_objective > safety > resilience > economy > sustainability > total_cost > plan_id"
    row["primary_indicator"] = "compromise_distance"
    row["primary_value"] = best_distance
    row["primary_ascending"] = True
    row["preference_weights"] = "、".join(f"{name}={raw_weights[idx]:.3f}" for idx, name in enumerate(OBJECTIVE_COLUMNS))
    return row


def recommend(pareto: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if pareto.empty:
        return pd.DataFrame()
    frame = pareto.copy()
    rows = [_pick_by_rule(frame, label) for label in CATEGORY_RULES]
    rows.append(_compromise_row(frame, weights))
    result = pd.DataFrame(rows)
    ordered = [
        "安全性最高代表方案",
        "经济性最高代表方案",
        "可持续性最高代表方案",
        "韧性最高代表方案",
        "综合折中代表方案",
    ]
    for column, default in {
        "preference_weights": "",
        "tie_candidate_ids": "",
        "tie_break_rule": "",
        "primary_indicator": "",
        "primary_value": 0.0,
        "primary_ascending": False,
        "compromise_distance": 0.0,
        "weakest_objective": 0.0,
    }.items():
        if column not in result.columns:
            result[column] = default
        result[column] = result[column].fillna(default)
    result["recommendation_type"] = pd.Categorical(result["recommendation_type"], categories=ordered, ordered=True)
    return result.sort_values("recommendation_type").reset_index(drop=True)
