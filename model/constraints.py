from __future__ import annotations

from .schemas import GovernanceUnit, SolverConfig


def check_plan_constraints(
    units: list[GovernanceUnit],
    details: list[dict[str, object]],
    total_cost: float,
    config: SolverConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if total_cost > config.total_budget + 1e-9:
        reasons.append(f"总成本{total_cost:.2f}超过预算{config.total_budget:.2f}")
    detail_by_id = {str(item["unit_id"]): item for item in details}
    for unit in units:
        detail = detail_by_id[unit.unit_id]
        if unit.risk >= 3 and not config.allow_untreated_medium_high and detail["measure_ids"] == "NONE":
            reasons.append(f"{unit.unit_name}为中高风险单元但未治理")
        if unit.risk >= 4 or unit.haz >= 4:
            if float(detail["fs_rain"]) < config.high_risk_min_fs:
                reasons.append(f"{unit.unit_name}降雨工况FS未达到{config.high_risk_min_fs:.2f}")
            if float(detail["safety"]) < config.high_risk_min_safety:
                reasons.append(f"{unit.unit_name}安全得分未达到{config.high_risk_min_safety:.1f}")
    return not reasons, reasons
