from __future__ import annotations

import json
import time
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pandas as pd

from .combinations import estimate_enumerations, generate_unit_options
from .pareto import pareto_front
from .recommendation import recommend
from .regional import evaluate_village_scheme
from .schemas import GovernanceUnit, Measure, SolveResult, SolverConfig, validate_measures, validate_units, validate_solver_config
from .spatial import validate_topology
from .screening import screen_all
from .tagging import in_governance_library, tagged_unit_record
from .i18n import FIELD_LABELS, chinese_frame


def configuration_hash(units: list[GovernanceUnit], measures: list[Measure], config: SolverConfig, method: str, topology=None) -> str:
    payload = {
        "units": [asdict(item) for item in units], "measures": [asdict(item) for item in measures],
        "config": asdict(config), "method": method, "topology": topology,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def run_identity(units: list[GovernanceUnit], measures: list[Measure], config: SolverConfig, method: str, topology=None) -> dict[str, object]:
    now = datetime.now(timezone.utc).astimezone()
    digest = configuration_hash(units, measures, config, method, topology)
    return {
        "scenario_id": f"SCN-{now:%Y%m%d-%H%M%S}-{digest[:6].upper()}",
        "run_time": now.isoformat(timespec="seconds"), "configuration_hash": digest,
        "random_seed": config.random_seed if method == "nsga2_discrete" else "不适用",
    }


def is_result_stale(last_input_hash: str | None, current_hash: str) -> bool:
    """输入未运行过或配置标识变化时，既有结果必须视为失效。"""
    return last_input_hash is None or last_input_hash != current_hash


def _finite_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return True
    return bool(numeric.notna().all().all()) and bool((numeric.abs() < float("inf")).all().all())


def solve_exact(
    units: list[GovernanceUnit],
    measures: list[Measure],
    config: SolverConfig | None = None,
    assumptions_path: str | Path | None = None,
    topology: dict[str, object] | None = None,
) -> SolveResult:
    started = time.perf_counter()
    config = config or SolverConfig()
    validate_solver_config(config)
    errors = validate_units(units) + validate_measures(measures)
    if errors:
        raise ValueError("\n".join(errors))
    active_units = [unit for unit in units if in_governance_library(unit)[0]]
    if not active_units:
        raise ValueError("没有治理单元命中入库规则，请检查Risk、L或识别/重点关注字段。")
    if topology is not None:
        validate_topology(topology, {u.unit_id for u in active_units})
    screening_results = screen_all(active_units, measures)
    screening_frame = pd.DataFrame([item.to_dict() for item in screening_results])
    tagged_frame = pd.DataFrame([tagged_unit_record(unit) for unit in units])

    options_by_unit = {
        unit.unit_id: generate_unit_options(unit, measures, screening_results, config)
        for unit in active_units
    }
    empty_units = [unit.unit_name for unit in active_units if not options_by_unit[unit.unit_id]]
    if empty_units:
        raise ValueError(f"以下单元没有满足组合约束的候选方案：{'、'.join(empty_units)}")
    enumeration_count = estimate_enumerations(options_by_unit)
    if enumeration_count > config.max_enumerations:
        raise ValueError(
            f"预计需枚举{enumeration_count:,}个村域组合，超过上限{config.max_enumerations:,}。"
            "请减少治理单元、降低每单元最大措施数或收紧候选规则。"
        )

    feasible_rows: list[dict[str, object]] = []
    detail_lookup: dict[str, list[dict[str, object]]] = {}
    rejected_budget = 0
    rejected_safety = 0
    best_gap: tuple[float, list[str]] | None = None
    rejection_reasons: dict[str, int] = {}
    option_lists = [options_by_unit[unit.unit_id] for unit in active_units]
    for selected_by_unit in product(*option_lists):
        scheme = evaluate_village_scheme(
            active_units,
            [tuple(selected) for selected in selected_by_unit],
            config,
            topology=topology,
        )
        details = scheme["unit_details"]
        region = scheme["regional_objectives"]
        valid = bool(scheme["feasible"])
        reasons = list(scheme["constraint_reasons"])
        if not valid:
            rejected_budget += int(any("预算" in reason for reason in reasons))
            rejected_safety += int(any("FS" in reason or "安全得分" in reason or "未治理" in reason for reason in reasons))
            for reason in reasons:
                category = "预算约束" if "预算" in reason else "安全底线或治理约束"
                rejection_reasons[category] = rejection_reasons.get(category, 0) + 1
            gap = len(reasons)
            if best_gap is None or gap < best_gap[0]:
                best_gap = (gap, reasons)
            continue
        plan_id = f"P{len(feasible_rows) + 1:05d}"
        configuration = "；".join(f"{item['unit_name']}={item['measures']}" for item in details)
        review = sorted({str(item["expert_review"]) for item in details if item["expert_review"] != "无"})
        row: dict[str, object] = {
            "plan_id": plan_id,
            **region,
            "total_cost": float(scheme["lifecycle_cost_total"]),
            "lci": float(scheme["regional_lci"]),
            "path_coverage": float(scheme["path_coverage"]),
            "coordination_score": float(scheme["coordination_score"]),
            "N_key": int(scheme["N_key"]),
            "N_covered": int(scheme["N_covered"]),
            "normalized_risk_weights": json.dumps(scheme["normalized_risk_weights"], ensure_ascii=False, sort_keys=True),
            "constraint_status": json.dumps(scheme["constraint_status"], ensure_ascii=False, sort_keys=True),
            "constraint_reasons": "；".join(reasons) if reasons else "无",
            "configuration_summary": configuration,
            "expert_review": "；".join(review) if review else "无",
        }
        for item in details:
            row[f"unit_{item['unit_id']}"] = item["measures"]
        feasible_rows.append(row)
        detail_lookup[plan_id] = details

    feasible = pd.DataFrame(feasible_rows)
    if feasible.empty:
        closest = "；".join(best_gap[1]) if best_gap else "未记录最接近可行方案。"
        raise ValueError(
            f"没有方案同时满足约束。预算约束剔除{rejected_budget}个；安全底线或治理约束剔除{rejected_safety}个；"
            f"最接近可行方案的主要差距：{closest}。建议优先检查预算上限、安全底线或中高风险单元治理要求。"
        )
    pareto = pareto_front(feasible)
    representatives = recommend(pareto, config.preference_weights)
    detail_rows: list[dict[str, object]] = []
    for _, recommendation in representatives.iterrows():
        for detail in detail_lookup[str(recommendation["plan_id"])]:
            detail_rows.append({
                "recommendation_type": recommendation["recommendation_type"],
                "plan_id": recommendation["plan_id"],
                **detail,
            })
    representative_details = pd.DataFrame(detail_rows)

    if assumptions_path and Path(assumptions_path).exists():
        assumptions = pd.read_csv(assumptions_path)
    else:
        assumptions = pd.DataFrame()
    for name, frame in {
        "治理单元标签": tagged_frame,
        "措施筛选": screening_frame,
        "可行方案": feasible,
        "Pareto方案": pareto,
        "代表方案": representatives,
        "代表方案明细": representative_details,
    }.items():
        if not _finite_frame(frame):
            raise RuntimeError(f"{name}包含NaN或无穷值。")

    runtime = time.perf_counter() - started
    metadata = {
        **run_identity(units, measures, config, "exact_enumeration", topology),
        "solver": "exact_enumeration",
        "deterministic": True,
        "active_units": len(active_units),
        "estimated_combinations": enumeration_count,
        "feasible_plans": len(feasible),
        "pareto_plans": len(pareto),
        "rejected_by_budget": rejected_budget,
        "rejected_by_safety_or_treatment": rejected_safety,
        "runtime_seconds": round(runtime, 6),
        "topology_paths": len(topology.get("paths", [])) if topology else 0,
        "config": {
            **asdict(config),
            "preference_weights": dict(config.preference_weights),
        },
    }
    return SolveResult(
        tagged_units=tagged_frame,
        measure_library=pd.DataFrame([measure.to_dict() for measure in measures]),
        screening=screening_frame,
        all_feasible=feasible,
        pareto=pareto,
        representatives=representatives,
        representative_details=representative_details,
        run_metadata=metadata,
        assumptions=assumptions,
        constraint_statistics=pd.DataFrame([
            {"约束类型": key, "剔除次数": value} for key, value in sorted(rejection_reasons.items())
        ]),
        detail_lookup=detail_lookup,
    )


def result_to_json(result: SolveResult) -> str:
    payload = {
        "运行元数据": result.run_metadata,
        "字段中英文映射": FIELD_LABELS,
        "治理单元及标签": chinese_frame(result.tagged_units).to_dict("records"),
        "工程措施库": chinese_frame(result.measure_library).to_dict("records"),
        "准入筛选结果": chinese_frame(result.screening).to_dict("records"),
        "全部可行方案": chinese_frame(result.all_feasible).to_dict("records"),
        "Pareto候选方案": chinese_frame(result.pareto).to_dict("records"),
        "代表方案": chinese_frame(result.representatives).to_dict("records"),
        "代表方案单元明细": chinese_frame(result.representative_details).to_dict("records"),
        "参数依据状态": result.assumptions.replace({"D": "待标定参数"}).to_dict("records"),
        "NSGA-II迭代历史": chinese_frame(result.iteration_history).to_dict("records"),
        "约束统计": result.constraint_statistics.to_dict("records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parameters_and_assumptions(result: SolveResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    config = dict(result.run_metadata.get("config", {}))
    preference = config.pop("preference_weights", {})
    for key, value in config.items():
        rows.append({
            "record_type": "当前运行参数",
            "parameter_or_rule": key,
            "current_value": value,
            "evidence_level": "运行配置",
            "purpose": "控制本次模型运行",
            "calibration_needed": "按项目需求配置",
        })
    for key, value in preference.items():
        rows.append({
            "record_type": "Pareto集推荐权重",
            "parameter_or_rule": f"preference_{key}",
            "current_value": value,
            "evidence_level": "用户偏好",
            "purpose": "仅用于Pareto集内部综合折中推荐",
            "calibration_needed": "由决策偏好设定",
        })
    for item in result.assumptions.to_dict("records"):
        rows.append({
            "record_type": "参数依据清单",
            "parameter_or_rule": item.get("parameter_or_rule", ""),
            "current_value": item.get("current_value", ""),
            "evidence_level": "待标定参数" if item.get("evidence_level", "D") == "D" else item.get("evidence_level", ""),
            "purpose": item.get("purpose", ""),
            "calibration_needed": item.get("calibration_needed", ""),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["parameter_or_rule"] = frame["parameter_or_rule"].map(lambda x: FIELD_LABELS.get(str(x), str(x)))
    return frame.rename(columns={
        "record_type":"记录类型", "parameter_or_rule":"参数或规则", "current_value":"当前值",
        "evidence_level":"依据状态", "purpose":"用途", "calibration_needed":"复核或标定要求",
    })


def solve(
    units: list[GovernanceUnit],
    measures: list[Measure],
    config: SolverConfig | None = None,
    method: str = "exact",
    assumptions_path: str | Path | None = None,
    topology: dict[str, object] | None = None,
) -> SolveResult:
    """统一求解器接口；默认稳定基线为精确枚举。"""
    if method == "exact":
        return solve_exact(units, measures, config, assumptions_path, topology=topology)
    if method == "nsga2":
        from .nsga2 import solve_nsga2
        return solve_nsga2(units, measures, config, assumptions_path, topology=topology)
    raise ValueError(f"未知求解模式：{method}")
