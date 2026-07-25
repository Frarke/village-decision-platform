from __future__ import annotations

from .schemas import GovernanceUnit


def normalized_risk_weights(units: list[GovernanceUnit]) -> dict[str, float]:
    raw = {unit.unit_id: max(0.0, unit.risk_weight) for unit in units}
    total = sum(raw.values())
    if total <= 0:
        equal = 1.0 / len(units) if units else 0.0
        return {unit.unit_id: equal for unit in units}
    return {unit_id: value / total for unit_id, value in raw.items()}


def evaluate_region(
    units: list[GovernanceUnit],
    details: list[dict[str, object]],
) -> dict[str, float]:
    if not details:
        raise ValueError("区域评价至少需要一个治理单元。")
    weights = normalized_risk_weights(units)
    weighted = lambda key: sum(weights[str(item["unit_id"])] * float(item[key]) for item in details)
    safety_values = [float(item["safety"]) for item in details]
    return {
        "total_cost": round(sum(float(item["actual_cost"]) for item in details), 6),
        "safety": round(0.6 * min(safety_values) + 0.4 * weighted("safety"), 6),
        "economy": round(weighted("economy"), 6),
        "sustainability": round(weighted("sustainability"), 6),
        "resilience": round(weighted("resilience"), 6),
    }
