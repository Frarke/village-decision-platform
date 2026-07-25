from __future__ import annotations

import numpy as np
import pandas as pd


OBJECTIVE_COLUMNS = ["safety", "economy", "sustainability", "resilience"]


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """四目标全部为越大越好。"""
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    values = frame[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    frontier: list[int] = []
    for idx, point in enumerate(values):
        dominated = False
        survivors: list[int] = []
        for existing in frontier:
            incumbent = values[existing]
            if dominates(incumbent, point):
                dominated = True
                break
            if not dominates(point, incumbent):
                survivors.append(existing)
        if not dominated:
            survivors.append(idx)
            frontier = survivors
    result = frame.iloc[frontier].copy()
    result = result.drop_duplicates(subset=OBJECTIVE_COLUMNS, keep="first")
    return result.sort_values(["total_cost", "safety"], ascending=[True, False]).reset_index(drop=True)
