from __future__ import annotations

import numpy as np
import pandas as pd

from .pareto import OBJECTIVE_COLUMNS


LABELS = {
    "安全优先方案": "优先最大化区域安全性",
    "经济优先方案": "优先最大化经济性得分并兼顾实际成本",
    "生态友好方案": "优先最大化可持续性",
    "韧性优先方案": "优先最大化韧性",
    "综合折中方案": "在Pareto集内按偏好权重计算到理想点的归一化距离",
}


def recommend(pareto: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if pareto.empty:
        return pd.DataFrame()
    choices: list[tuple[str, int]] = []
    choices.append(("安全优先方案", int(pareto.sort_values(["safety", "economy"], ascending=[False, False]).index[0])))
    choices.append(("经济优先方案", int(pareto.sort_values(["economy", "total_cost"], ascending=[False, True]).index[0])))
    choices.append(("生态友好方案", int(pareto.sort_values(["sustainability", "safety"], ascending=[False, False]).index[0])))
    choices.append(("韧性优先方案", int(pareto.sort_values(["resilience", "safety"], ascending=[False, False]).index[0])))

    matrix = pareto[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
    low = matrix.min(axis=0)
    span = matrix.max(axis=0) - low
    normalized = np.where(span > 0, (matrix - low) / np.where(span == 0, 1.0, span), 1.0)
    raw_weights = np.array([max(0.0, float(weights.get(name, 0.0))) for name in OBJECTIVE_COLUMNS])
    if raw_weights.sum() <= 0:
        raw_weights = np.ones(4)
    raw_weights /= raw_weights.sum()
    distances = np.sqrt(((1.0 - normalized) ** 2 * raw_weights).sum(axis=1))
    compromise_position = int(np.argmin(distances))
    choices.append(("综合折中方案", int(pareto.index[compromise_position])))

    rows: list[dict[str, object]] = []
    for category, idx in choices:
        row = pareto.loc[idx].to_dict()
        row = {"recommendation_type": category, "recommendation_basis": LABELS[category], **row}
        rows.append(row)
    return pd.DataFrame(rows)
