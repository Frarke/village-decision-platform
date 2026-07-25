from __future__ import annotations

from statistics import fmean

from .schemas import GovernanceUnit, Measure, SolverConfig


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _terrain_cost_factor(unit: GovernanceUnit) -> float:
    factor = 1.0
    if unit.slope_deg > 45:
        factor += 0.08
    if unit.height_m > 20:
        factor += 0.07
    if not unit.accessible:
        factor += 0.12
    return factor


def _combined_delta_fs(measures: tuple[Measure, ...], config: SolverConfig) -> tuple[float, float]:
    if not measures:
        return 0.0, 1.0
    # 输移拦截和承灾体防护类工程主要削减运动后果，不直接提高物源斜坡稳定系数。
    consequence_control = {"谷坊坝", "柔性拦挡结构"}
    ordered = sorted(
        (measure.delta_fs for measure in measures if measure.measure_name not in consequence_control),
        reverse=True,
    )
    if not ordered:
        return 0.0, 1.0
    delta = ordered[0] + sum(value * config.general_second_measure_factor for value in ordered[1:])
    names = {measure.measure_name for measure in measures}
    synergy = 1.0
    if "截排水工程" in names and names.intersection({"灌草种植", "植被混凝土", "格构锚固"}):
        synergy = config.drainage_synergy
    return min(0.70, delta * synergy), synergy


def evaluate_unit(
    unit: GovernanceUnit,
    measures: tuple[Measure, ...],
    config: SolverConfig,
) -> dict[str, object]:
    delta_fs, synergy = _combined_delta_fs(measures, config)
    fs_rain = unit.fs0 * unit.k_rain + delta_fs
    # 待标定映射：FS=0.80映射为0分，FS=1.40映射为100分，区间内线性插值。
    s_fs = clamp((fs_rain - 0.80) / 0.60 * 100.0)
    chain_base = 20.0
    chain_score = clamp(chain_base + sum(measure.chain_delta for measure in measures))
    safety = clamp(0.6 * s_fs + 0.4 * chain_score)

    if measures:
        lci = fmean(
            0.65 * measure.build_level
            + 0.20 * measure.maintain_level
            + 0.15 * measure.construct_level
            for measure in measures
        )
        lci_score = clamp(100.0 - 20.0 * (lci - 1.0))
        eco = fmean(measure.eco_score for measure in measures)
        carbon = fmean(100.0 - 20.0 * (measure.carbon_level - 1.0) for measure in measures)
        disturb = fmean(100.0 - 20.0 * (measure.disturbance_level - 1.0) for measure in measures)
        if unit.slope_deg > 45:
            disturb -= 5.0
        sustainability = clamp(0.4 * eco + 0.4 * carbon + 0.2 * clamp(disturb))
        function_count = len({measure.function_type for measure in measures})
        recovery_difficulty = fmean(measure.recovery_difficulty for measure in measures)
        covered = min(unit.key_paths, sum(measure.covered_paths for measure in measures))
    else:
        lci = 1.0
        lci_score = 100.0
        # 不治理虽无新增工程投入，但不能因此获得较高生态/韧性表现。
        sustainability = 45.0
        function_count = 0
        recovery_difficulty = 3.0
        covered = 0

    redundancy = min(100.0, (15.0 if not measures else 40.0) + 15.0 * function_count)
    recover = clamp(100.0 - 20.0 * (recovery_difficulty - 1.0))
    coverage = (100.0 if unit.key_paths == 0 else 100.0 * covered / unit.key_paths) if measures else 0.0
    resilience = clamp(0.5 * redundancy + 0.3 * recover + 0.2 * coverage)
    # 全寿命周期成本：基础建设、维护折现、施工组织和地形修正均显式列示。
    # 维护与组织比例为待标定参数，保留报告LCI口径作为经济性基础得分。
    construction_cost = sum(measure.actual_cost for measure in measures)
    maintenance_npv = sum(
        measure.actual_cost * (0.03 + 0.015 * (measure.maintain_level - 1.0))
        for measure in measures
    )
    organization_cost = sum(
        measure.actual_cost * 0.02 * (measure.construct_level - 1.0)
        for measure in measures
    )
    cost_factor = _terrain_cost_factor(unit)
    subtotal = construction_cost + maintenance_npv + organization_cost
    terrain_adjustment_cost = subtotal * (cost_factor - 1.0)
    lifecycle_cost = subtotal + terrain_adjustment_cost
    # 保留科技报告LCI评分，并以实际全寿命周期成本作为主导校正项。
    # 这样既保留建设/维护/施工等级信息，也避免组合措施的等级均值掩盖新增成本。
    lifecycle_cost_score = clamp(100.0 - 0.35 * lifecycle_cost)
    economy = clamp(0.20 * lci_score + 0.80 * lifecycle_cost_score)

    review_items: list[str] = []
    if any(measure.measure_name == "抗滑桩" for measure in measures) and not unit.slip_surface_confirmed:
        review_items.append("抗滑桩需勘察复核滑面及稳定嵌固段")
    if any(measure.evidence_level.upper() == "D" or "待标定" in measure.evidence_level for measure in measures):
        review_items.append("措施效应和成本为待标定参数")

    return {
        "unit_id": unit.unit_id,
        "unit_name": unit.unit_name,
        "measures": "＋".join(measure.measure_name for measure in measures) if measures else "不治理",
        "measure_ids": "+".join(measure.measure_id for measure in measures) if measures else "NONE",
        "actual_cost": round(lifecycle_cost, 6),
        "construction_cost": round(construction_cost, 6),
        "maintenance_npv": round(maintenance_npv, 6),
        "organization_cost": round(organization_cost, 6),
        "terrain_adjustment_cost": round(terrain_adjustment_cost, 6),
        "lifecycle_cost": round(lifecycle_cost, 6),
        "cost_factor": round(cost_factor, 4),
        "fs_rain": round(fs_rain, 6),
        "fs_before_rain": round(unit.fs0 * unit.k_rain, 6),
        "delta_fs": round(delta_fs, 6),
        "synergy_factor": round(synergy, 4),
        "s_fs": round(s_fs, 6),
        "s_chain": round(chain_score, 6),
        "safety": round(safety, 6),
        "lci": round(lci, 6),
        "lci_score": round(lci_score, 6),
        "lifecycle_cost_score": round(lifecycle_cost_score, 6),
        "economy": round(economy, 6),
        "sustainability": round(sustainability, 6),
        "resilience": round(resilience, 6),
        "function_count": function_count,
        "covered_paths": covered,
        "key_paths": unit.key_paths,
        "expert_review": "；".join(review_items) if review_items else "无",
        "residual_hazard": max(1, min(5, int(round(unit.haz - delta_fs * 4.0 - chain_score / 70.0)))),
        "residual_risk": max(1, min(5, int(round(unit.risk - delta_fs * 3.0 - chain_score / 80.0)))),
        "risk_reduction": round(max(0.0, unit.risk - max(1.0, unit.risk - delta_fs * 3.0 - chain_score / 80.0)), 3),
        "downstream_transfer_review": "需复核导排出口与下游承灾体" if any(m.measure_name == "截排水工程" for m in measures) else "未识别新增导排风险",
    }
