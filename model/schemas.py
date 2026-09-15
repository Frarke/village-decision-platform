from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from math import isfinite
from typing import Any

import pandas as pd
from .parameters import EvaluationParameters


BOOL_COLUMNS_UNITS = [
    "identified_or_risk_area",
    "focus_unit",
    "safe_drainage_outlet",
    "artificial_or_bare",
    "anchor_condition",
    "slip_surface_or_review",
    "slip_surface_confirmed",
    "motion_path",
    "accessible",
    "wall_back_drainage",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是", "有"}


def _clean_text(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


@dataclass(frozen=True)
class GovernanceUnit:
    unit_id: str
    unit_name: str
    risk: int
    haz: int
    distance_m: float
    rainfall_mm: float
    slope_deg: float
    height_m: float
    catchment_km2: float
    gully_level: int
    foot_space_m: float
    fs0: float
    k_rain: float
    identified_or_risk_area: bool
    focus_unit: bool
    safe_drainage_outlet: bool
    artificial_or_bare: bool
    anchor_condition: bool
    slip_surface_or_review: bool
    slip_surface_confirmed: bool
    motion_path: bool
    accessible: bool
    wall_back_drainage: bool
    key_paths: int
    risk_weight: float
    evidence_level: str = "待标定参数"
    notes: str = ""

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> "GovernanceUnit":
        values = dict(row)
        for name in BOOL_COLUMNS_UNITS:
            values[name] = _as_bool(values.get(name, False))
        values["unit_id"] = _clean_text(values.get("unit_id", ""))
        values["unit_name"] = _clean_text(values.get("unit_name", ""))
        values["evidence_level"] = _clean_text(values.get("evidence_level", ""), "待校验")
        values["notes"] = _clean_text(values.get("notes", ""))
        int_fields = {"risk", "haz", "gully_level", "key_paths"}
        for name in int_fields:
            values[name] = int(float(values[name]))
        float_fields = {
            "distance_m", "rainfall_mm", "slope_deg", "height_m",
            "catchment_km2", "foot_space_m", "fs0", "k_rain", "risk_weight",
        }
        for name in float_fields:
            values[name] = float(values[name])
        return cls(**{key: values[key] for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Measure:
    measure_id: str
    measure_name: str
    function_type: str
    applicable_objects: str
    actual_cost: float
    build_level: float
    maintain_level: float
    construct_level: float
    delta_fs: float
    chain_delta: float
    eco_score: float
    carbon_level: float
    disturbance_level: float
    recovery_difficulty: float
    covered_paths: int
    is_primary_control: bool
    evidence_level: str = "待标定参数"
    notes: str = ""

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> "Measure":
        values = dict(row)
        values["is_primary_control"] = _as_bool(values.get("is_primary_control", False))
        for name in ["measure_id", "measure_name", "function_type", "applicable_objects"]:
            values[name] = _clean_text(values.get(name, ""))
        values["evidence_level"] = _clean_text(values.get("evidence_level", ""), "待校验")
        values["notes"] = _clean_text(values.get("notes", ""))
        values["covered_paths"] = int(float(values["covered_paths"]))
        for name in [
            "actual_cost", "build_level", "maintain_level", "construct_level",
            "delta_fs", "chain_delta", "eco_score", "carbon_level",
            "disturbance_level", "recovery_difficulty",
        ]:
            values[name] = float(values[name])
        return cls(**{key: values[key] for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScreeningResult:
    unit_id: str
    unit_name: str
    measure_id: str
    measure_name: str
    eligible: bool
    entry_conditions: str
    failed_restrictions: str
    exclusion_reason: str
    combination_required: str
    review_required: str
    evidence_level: str

    def to_dict(self) -> dict[str, Any]:
        return {"screening_id": f"{self.unit_id}-{self.measure_id}", **asdict(self)}


@dataclass
class SolverConfig:
    total_budget: float = 520.0
    max_measures_per_unit: int = 2
    allow_untreated_medium_high: bool = False
    high_risk_min_fs: float = 1.05
    high_risk_min_safety: float = 58.0
    max_enumerations: int = 200_000
    drainage_synergy: float = 1.10
    general_second_measure_factor: float = 0.90
    nsga_population_size: int = 60
    nsga_generations: int = 25
    nsga_crossover_probability: float = 0.90
    nsga_mutation_probability: float = 0.20
    random_seed: int = 42
    preference_weights: dict[str, float] = field(default_factory=lambda: {
        "safety": 0.35, "economy": 0.20, "sustainability": 0.20, "resilience": 0.25
    })
    evaluation_parameters: EvaluationParameters = field(default_factory=EvaluationParameters)


@dataclass
class SolveResult:
    tagged_units: pd.DataFrame
    measure_library: pd.DataFrame
    screening: pd.DataFrame
    all_feasible: pd.DataFrame
    pareto: pd.DataFrame
    representatives: pd.DataFrame
    representative_details: pd.DataFrame
    run_metadata: dict[str, Any]
    assumptions: pd.DataFrame
    iteration_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    constraint_statistics: pd.DataFrame = field(default_factory=pd.DataFrame)
    detail_lookup: dict[str, list[dict[str, object]]] = field(default_factory=dict)


def load_units(path: str | Path) -> list[GovernanceUnit]:
    frame = pd.read_csv(path)
    return [GovernanceUnit.from_record(record) for record in frame.to_dict("records")]


def load_measures(path: str | Path) -> list[Measure]:
    frame = pd.read_csv(path)
    return [Measure.from_record(record) for record in frame.to_dict("records")]


def validate_units(units: list[GovernanceUnit]) -> list[str]:
    errors: list[str] = []
    if not units:
        return ["治理单元不能为空。"]
    ids = [u.unit_id for u in units]
    if len(ids) != len(set(ids)):
        errors.append("治理单元编号必须唯一。")
    for unit in units:
        if not unit.unit_id.strip():
            errors.append("治理单元编号不能为空。")
        if any(not isfinite(value) for value in unit.to_dict().values() if isinstance(value, (int, float))):
            errors.append(f"{unit.unit_name}：数值不能含NaN或无穷值。")
        if not 1 <= unit.risk <= 5 or not 1 <= unit.haz <= 5:
            errors.append(f"{unit.unit_name}：Risk和Haz必须在1～5之间。")
        if min(unit.distance_m, unit.rainfall_mm, unit.height_m, unit.catchment_km2, unit.foot_space_m) < 0:
            errors.append(f"{unit.unit_name}：距离、降雨、坡高和汇流面积不得为负。")
        if not 0 <= unit.slope_deg <= 90:
            errors.append(f"{unit.unit_name}：坡度必须在0～90°之间。")
        if unit.fs0 <= 0 or not 0 < unit.k_rain <= 1.5:
            errors.append(f"{unit.unit_name}：FS0与K_rain取值无效。")
        if unit.risk_weight < 0 or unit.key_paths < 0:
            errors.append(f"{unit.unit_name}：风险权重和关键风险路径数不得为负。")
    if sum(unit.risk_weight for unit in units) <= 0:
        errors.append("风险权重必须显式提供且总和大于零，不得自动替换为均等权重。")
    return errors


def validate_measures(measures: list[Measure]) -> list[str]:
    errors: list[str] = []
    if not measures:
        return ["措施库不能为空。"]
    ids = [m.measure_id for m in measures]
    if len(ids) != len(set(ids)):
        errors.append("措施编号必须唯一。")
    for measure in measures:
        if not measure.measure_id.strip():
            errors.append("措施编号不能为空。")
        if any(not isfinite(value) for value in measure.to_dict().values() if isinstance(value, (int, float))):
            errors.append(f"{measure.measure_name}：数值不能含NaN或无穷值。")
        if measure.chain_delta < 0 or measure.covered_paths < 0:
            errors.append(f"{measure.measure_name}：效果和路径数不得为负。")
        if measure.actual_cost < 0 or measure.delta_fs < 0:
            errors.append(f"{measure.measure_name}：成本和稳定性提升值不得为负。")
        for field_name in ["build_level", "maintain_level", "construct_level", "carbon_level", "disturbance_level", "recovery_difficulty"]:
            value = getattr(measure, field_name)
            if not 1 <= value <= 5:
                errors.append(f"{measure.measure_name}：{field_name}必须在1～5之间。")
        if not 0 <= measure.eco_score <= 100:
            errors.append(f"{measure.measure_name}：生态得分必须在0～100之间。")
    return errors


def validate_solver_config(config: SolverConfig) -> None:
    for name in ("total_budget", "high_risk_min_fs", "high_risk_min_safety", "drainage_synergy",
                 "general_second_measure_factor", "nsga_crossover_probability", "nsga_mutation_probability"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"{name}须为有限非负数。")
    for name, low, high in (("max_measures_per_unit", 1, 8), ("max_enumerations", 1, 200_000),
                            ("nsga_population_size", 20, 200), ("nsga_generations", 1, 200),
                            ("random_seed", 0, 2**32 - 1)):
        value = getattr(config, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name}须为{low}～{high}整数。")
    if max(config.nsga_crossover_probability, config.nsga_mutation_probability) > 1:
        raise ValueError("交叉/变异概率须在0～1之间。")
    if config.high_risk_min_safety > 100 or config.high_risk_min_fs <= 0:
        raise ValueError("安全分数底线须在0～100之间，稳定系数底线须大于零。")
    if type(config.allow_untreated_medium_high) is not bool:
        raise ValueError("allow_untreated_medium_high须为布尔值。")
    names = {"safety", "economy", "sustainability", "resilience"}
    if not isinstance(config.preference_weights, dict) or set(config.preference_weights) != names:
        raise ValueError("偏好权重须包含完整的四目标键。")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0
           for v in config.preference_weights.values()) or sum(config.preference_weights.values()) <= 0:
        raise ValueError("偏好权重须有限非负且总和大于零。")
