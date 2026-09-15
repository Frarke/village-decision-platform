from __future__ import annotations

from .constraints import check_plan_constraints
from .combinations import validate_unit_combination
from .screening import screen_measure
from math import isfinite
from .evaluation import evaluate_unit
from .schemas import GovernanceUnit
from .schemas import Measure, SolverConfig
from .spatial import evaluate_path_coverage, validate_topology


def normalized_risk_weights(units: list[GovernanceUnit]) -> dict[str, float]:
    raw = {unit.unit_id: unit.risk_weight for unit in units}
    if any(not isfinite(v) or v < 0 for v in raw.values()):
        raise ValueError("风险权重须为有限非负数。")
    maximum = max(raw.values(), default=0.0)
    if maximum <= 0:
        raise ValueError("风险权重总和须大于零，不自动生成权重。")
    scaled = {unit_id: value / maximum for unit_id, value in raw.items()}
    total = sum(scaled.values())
    return {unit_id: value / total for unit_id, value in scaled.items()}


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
    if not units or len(units) != len(selected_by_unit):
        raise ValueError("方案编码长度须与非空治理单元列表一致。")
    ids = {unit.unit_id for unit in units}
    if len(ids) != len(units):
        raise ValueError("治理单元编号必须唯一。")
    if topology is not None:
        validate_topology(topology, ids)
    admission_reasons = []
    for unit, selected in zip(units, selected_by_unit, strict=True):
        if len({m.measure_id for m in selected}) != len(selected):
            admission_reasons.append(f"{unit.unit_id}存在重复措施")
        valid, reason = validate_unit_combination(unit, selected, config)
        if not valid:
            admission_reasons.append(f"{unit.unit_id}：{reason}")
        for measure in selected:
            screening = screen_measure(unit, measure)
            if not screening.eligible:
                admission_reasons.append(f"{unit.unit_id}措施{measure.measure_name}未准入：{screening.exclusion_reason}")
    details = [
        evaluate_unit(unit, measures, config)
        for unit, measures in zip(units, selected_by_unit, strict=True)
    ]
    region = evaluate_region(units, details, topology=topology)
    if topology is not None:
        # 单元局部覆盖与村域完整覆盖均读取同一条路径记录；局部只统计本单元承担的环节。
        for detail in details:
            uid = detail["unit_id"]
            related = [p for p in region["path_details"] if uid in
                       (p["source_unit"], p["transport_unit"], p["receptor_unit"])]
            controlled = sum(all(p[f"{stage}_controlled"] for stage in ("source", "transport", "receptor")
                                 if p[f"{stage}_unit"] == uid) for p in related)
            coverage = 100.0 * controlled / len(related) if related else 100.0
            if detail["measure_ids"] == "NONE":
                coverage = 0.0
            detail["key_paths"] = len(related)
            detail["covered_paths"] = controlled
            detail["coverage_score_local"] = round(coverage, 6)
            detail["resilience"] = round(0.5 * detail["redundancy_score"] +
                                         0.3 * detail["recovery_score"] + 0.2 * coverage, 6)
    feasible, reasons = check_plan_constraints(units, details, float(region["total_cost"]), config)
    reasons = admission_reasons + reasons
    feasible = feasible and not admission_reasons
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
            "admission_and_combination": not admission_reasons,
            "budget": not any("预算" in reason for reason in reasons),
            "safety_floor": not any("FS" in reason or "安全得分" in reason for reason in reasons),
            "treatment_rule": not any("未治理" in reason for reason in reasons),
        },
        "feasible": feasible,
    }
