from __future__ import annotations

from .constraints import check_plan_constraints
from .evaluation import evaluate_unit
from .schemas import GovernanceUnit
from .schemas import Measure, SolverConfig
from .spatial import evaluate_path_coverage


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
    topology: dict[str, object] | None = None,
) -> dict[str, float]:
    if not details:
        raise ValueError("区域评价至少需要一个治理单元。")
    weights = normalized_risk_weights(units)
    weighted = lambda key, default=0.0: sum(weights[str(item["unit_id"])] * float(item.get(key, default)) for item in details)
    safety_values = [float(item["safety"]) for item in details]
    total_cost = round(sum(float(item["actual_cost"]) for item in details), 6)
    result = {
        "total_cost": round(sum(float(item["actual_cost"]) for item in details), 6),
        "safety": round(0.6 * min(safety_values) + 0.4 * weighted("safety"), 6),
        "economy": round(weighted("economy"), 6),
        "sustainability": round(weighted("sustainability"), 6),
        "resilience": round(weighted("resilience"), 6),
        "lci": round(weighted("lci", 1.0), 6),
        "normalized_risk_weights": weights,
    }
    if topology is not None:
        path_summary = evaluate_path_coverage(topology, details)
        result["resilience"] = round(
            0.5 * weighted("redundancy_score")
            + 0.3 * weighted("recovery_score")
            + 0.2 * float(path_summary["coverage_score"]),
            6,
        )
        result["path_coverage"] = round(float(path_summary["coverage_score"]), 6)
        result["coordination_score"] = round(float(path_summary["coordination_score"]), 6)
        result["N_key"] = int(path_summary["N_key"])
        result["N_covered"] = int(path_summary["N_covered"])
        result["path_details"] = path_summary["path_details"]
        result["weak_links"] = path_summary["weak_links"]
        result["average_residual_risk"] = round(float(path_summary["average_residual_risk"]), 6)
        result["source_rate"] = round(float(path_summary["source_rate"]), 6)
        result["transport_rate"] = round(float(path_summary["transport_rate"]), 6)
        result["receptor_rate"] = round(float(path_summary["receptor_rate"]), 6)
    result["total_cost"] = total_cost
    return result


def evaluate_village_scheme(
    units: list[GovernanceUnit],
    selected_by_unit: list[tuple[Measure, ...]],
    config: SolverConfig,
    topology: dict[str, object] | None = None,
) -> dict[str, object]:
    details = [
        evaluate_unit(unit, measures, config)
        for unit, measures in zip(units, selected_by_unit, strict=True)
    ]
    region = evaluate_region(units, details, topology=topology)
    feasible, reasons = check_plan_constraints(units, details, float(region["total_cost"]), config)
    return {
        "unit_details": details,
        "regional_objectives": {
            "safety": float(region["safety"]),
            "economy": float(region["economy"]),
            "sustainability": float(region["sustainability"]),
            "resilience": float(region["resilience"]),
        },
        "regional_lci": float(region.get("lci", 0.0)),
        "normalized_risk_weights": dict(region.get("normalized_risk_weights", {})),
        "N_key": int(region.get("N_key", 0)),
        "N_covered": int(region.get("N_covered", 0)),
        "path_details": list(region.get("path_details", [])),
        "path_coverage": float(region.get("path_coverage", 0.0)),
        "coordination_score": float(region.get("coordination_score", 0.0)),
        "weak_links": list(region.get("weak_links", [])),
        "source_rate": float(region.get("source_rate", 0.0)),
        "transport_rate": float(region.get("transport_rate", 0.0)),
        "receptor_rate": float(region.get("receptor_rate", 0.0)),
        "average_residual_risk": float(region.get("average_residual_risk", 0.0)),
        "lifecycle_cost_total": float(region["total_cost"]),
        "constraint_reasons": reasons,
        "constraint_status": {
            "budget": not any("预算" in reason for reason in reasons),
            "safety_floor": not any("FS" in reason or "安全得分" in reason for reason in reasons),
            "treatment_rule": not any("未治理" in reason for reason in reasons),
        },
        "feasible": feasible,
    }
