"""供API、页面、批处理共用的空间决策服务；不依赖Streamlit。"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .combinations import generate_unit_options
from .data_adapter import _number
from .governance import SpatialConfig, build_governance
from .parameters import EvaluationParameters
from .pareto import OBJECTIVE_COLUMNS
from .regional import evaluate_village_scheme, normalized_risk_weights
from .schemas import (BOOL_COLUMNS_UNITS, GovernanceUnit, Measure, SolverConfig,
                      load_measures, validate_measures, validate_solver_config, validate_units)
from .screening import screen_all
from .solver import solve
from .spatial import validate_topology
from .spatial_binding import bind_spatial_context
from .engineering_symbols import measure_placements


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"
REPRESENTATIVE_LABELS = {
    "综合折中代表方案": "综合均衡", "安全性最高代表方案": "安全优先",
    "经济性最高代表方案": "经济优先", "韧性最高代表方案": "韧性优先",
    "可持续性最高代表方案": "可持续优先",
}
AUTOMATIC_FIELDS = {"unit_id", "risk", "haz", "distance_m", "slope_deg", "key_paths", "risk_weight",
                    "identified_or_risk_area", "focus_unit"}
PROFILE_FIELDS = set(GovernanceUnit.__dataclass_fields__) - AUTOMATIC_FIELDS
REQUIRED_PROFILE = PROFILE_FIELDS - {"unit_name", "evidence_level", "notes"}


def input_hash(request: dict) -> str:
    try:
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("输入必须为有限数值组成的JSON对象。") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_sample_request() -> dict:
    return json.loads((ROOT / "data/v1/sample_request.json").read_text(encoding="utf-8"))


def load_geographic_sample() -> dict:
    return json.loads((ROOT / "data/v1_1/peitian_request.json").read_text(encoding="utf-8"))


def _request_config(request):
    if not isinstance(request, dict):
        raise ValueError("请求须为JSON对象。")
    allowed = {"schema_version", "village", "spatial_layers", "engineering_profiles", "unit_risk_weights",
               "topology", "configuration", "measures", "spatial_bindings"}
    if set(request) - allowed:
        raise ValueError(f"未知请求字段：{', '.join(sorted(set(request) - allowed))}")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version须为1.0。")
    village = request.get("village")
    if not isinstance(village, dict) or not all(village.get(k) for k in ("id", "name", "source")):
        raise ValueError("village须提供id、name和source。")
    if village.get("data_status") not in {"MOCK", "REAL"}:
        raise ValueError("village.data_status须明确为MOCK或REAL。")
    config = request.get("configuration", {})
    if not isinstance(config, dict) or set(config) - {"spatial", "solver", "method"}:
        raise ValueError("configuration只支持spatial、solver、method。")
    try:
        spatial = SpatialConfig(**config.get("spatial", {}))
        raw_solver = dict(config.get("solver", {}))
        raw_parameters = raw_solver.pop("evaluation_parameters", {})
        solver = SolverConfig(**raw_solver, evaluation_parameters=EvaluationParameters(**raw_parameters))
        validate_solver_config(solver)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"配置字段或类型不合法：{exc}") from exc
    method = config.get("method", "nsga2")
    if method not in {"exact", "nsga2"}:
        raise ValueError("method仅支持exact或nsga2。")
    return spatial, solver, method


def prepare_request(request: dict) -> dict:
    digest = input_hash(request)
    spatial, _, _ = _request_config(request)
    prepared = build_governance(request["village"], request.get("spatial_layers"), spatial)
    prepared["schema_version"] = SCHEMA_VERSION
    prepared["input_hash"] = digest
    prepared["required_risk_weight_ids"] = [f["properties"]["unit_id"] for f in prepared["governance_units"]["features"]]
    prepared["status"] = "prepared" if prepared["required_risk_weight_ids"] else "no_governance_units"
    if prepared["required_risk_weight_ids"] and request.get("spatial_bindings") is not None:
        _, _, trace = bind_spatial_context(request["spatial_bindings"], prepared["governance_units"])
        prepared["spatial_binding_trace"] = trace
        prepared["missing_risk_weight_ids"] = []
    else:
        prepared["missing_risk_weight_ids"] = prepared["required_risk_weight_ids"]
    return prepared


def _model_units(request, prepared):
    profiles = request.get("engineering_profiles", {})
    weights = request.get("unit_risk_weights", {})
    ids = set(prepared["required_risk_weight_ids"])
    if not isinstance(weights, dict) or set(weights) != ids:
        raise ValueError(f"unit_risk_weights须显式提供且仅包含生成单元：{', '.join(sorted(ids))}；可先调用/prepare。")
    if not isinstance(profiles, dict):
        raise ValueError("engineering_profiles须为按profile_id索引的对象。")
    units = []
    for feature in prepared["governance_units"]["features"]:
        props = feature["properties"]
        profile = profiles.get(props["profile_id"])
        if not isinstance(profile, dict):
            raise ValueError(f"缺少工程调查属性区{props['profile_id']}。")
        missing = REQUIRED_PROFILE - set(profile)
        if missing or set(profile) - PROFILE_FIELDS:
            raise ValueError(f"属性区{props['profile_id']}字段错误；缺少{sorted(missing)}；不允许{sorted(set(profile)-PROFILE_FIELDS)}。")
        for name in REQUIRED_PROFILE:
            value = profile[name]
            if name in BOOL_COLUMNS_UNITS:
                if type(value) is not bool:
                    raise ValueError(f"属性区{props['profile_id']}.{name}须显式提供布尔调查结果。")
            else:
                _number(value, f"属性区{props['profile_id']}.{name}", 0,
                        integer=name == "gully_level")
        weight = _number(weights[props["unit_id"]], f"{props['unit_id']}.risk_weight", 0)
        raw = {**profile, **{key: props[key] for key in ("unit_id", "risk", "haz", "distance_m", "slope_deg")},
               "unit_name": f"{props['unit_id']} {profile.get('unit_name', '治理单元')}",
               "identified_or_risk_area": True, "focus_unit": False, "key_paths": 0,
               "risk_weight": weight, "evidence_level": profile.get("evidence_level", "待校核调查参数"),
               "notes": profile.get("notes", "")}
        unit = GovernanceUnit.from_record(raw)
        units.append(unit)
    errors = validate_units(units)
    if errors:
        raise ValueError("；".join(errors))
    return units


def _measures(request):
    if "measures" not in request:
        return load_measures(ROOT / "data/default_measures.csv"), "内置八类措施库；效应与成本待标定"
    if not isinstance(request["measures"], list) or not 1 <= len(request["measures"]) <= 32:
        raise ValueError("measures须包含1～32条措施记录。")
    measures = []
    try:
        for row in request["measures"]:
            if type(row.get("is_primary_control")) is not bool:
                raise ValueError("措施is_primary_control须为布尔值。")
            for name in ("actual_cost", "build_level", "maintain_level", "construct_level", "delta_fs",
                         "chain_delta", "eco_score", "carbon_level", "disturbance_level", "recovery_difficulty", "covered_paths"):
                _number(row[name], f"措施.{name}", 0, integer=name == "covered_paths")
            measures.append(Measure.from_record(row))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"措施字段缺失或类型错误：{exc}") from exc
    errors = validate_measures(measures)
    if errors:
        raise ValueError("；".join(errors))
    return measures, "调用方提供的措施库；以各记录evidence_level为准"


def model_context(request: dict, prepared: dict | None = None):
    prepared = prepared or prepare_request(request)
    _, config, method = _request_config(request)
    if request.get("spatial_bindings") is not None:
        if request.get("unit_risk_weights") or request.get("topology"):
            raise ValueError("空间绑定与按单元编号的权重/路径不可同时提供，避免冲突。")
        weights, topology, _ = bind_spatial_context(request["spatial_bindings"], prepared["governance_units"])
        request = {**request, "unit_risk_weights": weights, "topology": topology}
    units = _model_units(request, prepared)
    topology = request.get("topology")
    validate_topology(topology, {u.unit_id for u in units})
    if not topology["paths"] and topology.get("status") != "confirmed_empty":
        raise ValueError("paths为空时须显式提供status=confirmed_empty，表示已复核无关键路径。")
    if not topology.get("source") or topology.get("data_status") not in {"MOCK", "REAL"}:
        raise ValueError("topology须声明source和data_status，人工路径必须标为MOCK。")
    if request["village"]["data_status"] == "REAL" and topology["data_status"] == "MOCK":
        raise ValueError("真实研究区不能混入未标明的模拟路径，请统一请求的数据性质或提供真实拓扑。")
    units = [replace(unit, key_paths=sum(unit.unit_id in (p["source_unit"], p["transport_unit"], p["receptor_unit"])
                                        for p in topology["paths"] if p.get("key", True))) for unit in units]
    measures, source = _measures(request)
    return prepared, units, measures, config, topology, method, source


def _records(frame: pd.DataFrame):
    return json.loads(frame.to_json(orient="records", force_ascii=False, double_precision=12))


def optimize_request(request: dict) -> dict:
    request = copy.deepcopy(request)
    prepared = prepare_request(request)
    if not prepared["required_risk_weight_ids"]:
        return {**prepared, "recommended_solution": None, "representative_solutions": [], "pareto_front": [],
                "map_result": {"boundary": prepared["boundary"], "solutions": {}},
                "summary": {"message": "当前规则与最小面积条件下没有生成治理单元；请检查覆盖范围及过滤统计。"},
                "metrics": prepared["spatial_metrics"], "provenance": {"village": request["village"]}}
    prepared, units, measures, config, topology, method, measure_source = model_context(request, prepared)
    result = solve(units, measures, config, method, ROOT / "data/assumptions.csv", topology=topology)
    screening = screen_all(units, measures)
    weights = normalized_risk_weights(units)
    for feature, unit in zip(prepared["governance_units"]["features"], units, strict=True):
        options = generate_unit_options(unit, measures, screening, config)
        feature["properties"].update({
            "unit_name": unit.unit_name, "risk_weight": unit.risk_weight,
            "normalized_risk_weight": weights[unit.unit_id],
            "available_measures": [r.measure_id for r in screening if r.unit_id == unit.unit_id and r.eligible],
            "candidate_combinations": [[m.measure_id for m in combo] for combo in options],
            "engineering_attributes": unit.to_dict(),
        })
    baseline = evaluate_village_scheme(units, [tuple() for _ in units], config, topology)
    baseline_by_id = {d["unit_id"]: d for d in baseline["unit_details"]}
    representatives, map_solutions = [], {}
    library = {m.measure_id: m for m in measures}
    for original_label, label in REPRESENTATIVE_LABELS.items():
        row = result.representatives[result.representatives.recommendation_type == original_label].iloc[0]
        plan_id = str(row["plan_id"])
        details = result.detail_lookup[plan_id]
        selected = [tuple(library[mid] for mid in d["measure_ids"].split("+") if mid != "NONE") for d in details]
        evaluated = evaluate_village_scheme(units, selected, config, topology)
        features = []
        for feature, detail in zip(prepared["governance_units"]["features"], details, strict=True):
            before = baseline_by_id[detail["unit_id"]]
            features.append({"type": "Feature", "geometry": feature["geometry"], "properties": {
                **feature["properties"], **detail, "plan_id": plan_id,
                "safety_before": before["safety"], "safety_change": round(detail["safety"] - before["safety"], 6),
                "resilience_before": before["resilience"],
                "resilience_change": round(detail["resilience"] - before["resilience"], 6),
                "data_status": request["village"]["data_status"],
            }})
        solution = {"plan_id": plan_id, "label": label, "objectives": evaluated["regional_objectives"],
                    "total_cost": evaluated["lifecycle_cost_total"], "basis": str(row["recommendation_basis"]),
                    "unit_details": details, "path_details": evaluated["path_details"],
                    "path_coverage": evaluated["path_coverage"], "constraint_status": evaluated["constraint_status"],
                    "configuration": [d["measure_ids"].split("+") if d["measure_ids"] != "NONE" else [] for d in details]}
        representatives.append(solution)
        map_solutions[label] = {"type": "FeatureCollection", "features": features,
                                "data_status": request["village"]["data_status"]}
    pareto = _records(result.pareto)
    for row in pareto:
        row["configuration"] = [d["measure_ids"].split("+") if d["measure_ids"] != "NONE" else []
                                for d in result.detail_lookup[row["plan_id"]]]
    notices = ["单元划分与空间统计为V1暂定规则，须现场复核。", "风险权重为显式输入，程序仅归一化。",
               "措施成本按每单元整套估算，未由面积推算工程量；拆分会影响成本。",
               "多措施组合、措施效应、成本和评分映射仍需工程标定。"]
    if request["village"]["data_status"] == "MOCK":
        notices.insert(0, "模拟示例空间数据及人工风险路径，不能作为真实工程研究成果。")
    if not any(p.get("key", True) for p in topology["paths"]):
        notices.append("已声明无关键路径；沿用暂定的空路径覆盖100分约定，不代表防控有效性验证。")
    payload = {
        "schema_version": SCHEMA_VERSION, "status": "completed", "input_hash": prepared["input_hash"],
        "recommended_solution": representatives[0], "representative_solutions": representatives,
        "pareto_front": pareto, "governance_units": prepared["governance_units"],
        "priority_areas": prepared["priority_areas"], "evaluation_areas": prepared["evaluation_areas"],
        "map_result": {"crs": "EPSG:4326", "boundary": prepared["boundary"], "solutions": map_solutions,
                       "measure_placements": measure_placements(map_solutions, prepared["spatial_metrics"]["metric_crs"]),
                       "placement_level": "治理单元配置示意；非施工设计点位、线位或工程量"},
        "summary": {"village_name": request["village"]["name"], "data_status": request["village"]["data_status"],
                    "governance_unit_count": len(units), "feasible_solutions": len(result.all_feasible),
                    "pareto_solutions": len(pareto), "notices": notices,
                    "baseline_objectives": baseline["regional_objectives"]},
        "metrics": {"spatial": prepared["spatial_metrics"], "optimization": result.run_metadata,
                    "iteration_history": _records(result.iteration_history)},
        "provenance": {"village": request["village"], "topology": topology,
                       "spatial_bindings": request.get("spatial_bindings"),
                       "spatial_binding_trace": prepared.get("spatial_binding_trace", []),
                       "measure_source": measure_source, "measures": [m.to_dict() for m in measures],
                       "solver_config": asdict(config), "spatial_config": prepared["spatial_metrics"]["configuration"],
                       "method_basis": "docs/METHOD_SPEC.md；科技报告表5-1至5-3",
                       "objective_directions": {name: "maximize" for name in OBJECTIVE_COLUMNS}},
        "screening": _records(result.screening),
    }
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    LOGGER.info("优化完成 input=%s method=%s units=%d pareto=%d", payload["input_hash"], method, len(units), len(pareto))
    return payload
