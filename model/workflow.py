"""引导式研判状态机、地图选择事件和输入标识的纯函数支持。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any, Mapping, Sequence

import pandas as pd

from .schemas import GovernanceUnit, Measure, SolverConfig, validate_measures, validate_units


GUIDED_STEPS = (
    "项目与研究区",
    "数据与治理单元",
    "工程措施准入",
    "约束与求解设置",
    "运行多目标优化",
    "Pareto方案比选",
    "空间协同研判",
    "复核与成果导出",
)

STEP_OUTPUTS = (
    "形成项目、村庄和数据版本信息",
    "形成通过校验的治理单元清单",
    "形成单元—措施准入判定矩阵",
    "形成可追溯的约束和算法配置",
    "形成可行方案与四目标Pareto候选集",
    "形成五类代表方案与推荐依据",
    "形成路径覆盖、薄弱环节和残余风险结论",
    "形成CSV、JSON和科研分析报告",
)


def optimization_hash(
    units: Sequence[GovernanceUnit],
    measures: Sequence[Measure],
    config: SolverConfig,
    method: str,
) -> str:
    """仅包含会改变可行/Pareto解集的输入；偏好权重不触发重算。"""
    config_data = asdict(config)
    config_data.pop("preference_weights", None)
    payload = {
        "units": [asdict(item) for item in units],
        "measures": [asdict(item) for item in measures],
        "config": config_data,
        "method": method,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def preference_hash(weights: Mapping[str, float]) -> str:
    return hashlib.sha256(
        json.dumps(dict(weights), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def extract_unit_id(event: Any, valid_ids: Sequence[str]) -> str | None:
    """从Streamlit/Plotly选择事件中稳定提取治理单元编号。"""
    valid = {str(item) for item in valid_ids}
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, Mapping):
        selection = event.get("selection", event)
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, Mapping):
        points = selection.get("points", [])
    for point in points or []:
        data = point.get("customdata") if isinstance(point, Mapping) else None
        candidates: list[Any] = []
        if isinstance(data, (list, tuple)):
            candidates.extend(data)
        elif data is not None:
            candidates.append(data)
        if isinstance(point, Mapping):
            candidates.extend([point.get("location"), point.get("text")])
        for candidate in candidates:
            value = str(candidate) if candidate is not None else ""
            if value in valid:
                return value
    return None


def synchronize_selected_unit(selected: str | None, valid_ids: Sequence[str]) -> str | None:
    ids = [str(item) for item in valid_ids]
    if not ids:
        return None
    return str(selected) if selected is not None and str(selected) in ids else ids[0]


def step_validation(
    step: int,
    *,
    project_name: str,
    village_name: str,
    units_df: pd.DataFrame,
    measures_df: pd.DataFrame,
    result_valid: bool,
) -> tuple[bool, list[str]]:
    """按真实校验结果判定步骤是否可完成，供UI和测试共用。"""
    errors: list[str] = []
    if step == 0:
        if not str(project_name).strip():
            errors.append("请填写当前项目名称。")
        if not str(village_name).strip():
            errors.append("请填写当前村庄或研究区名称。")
    elif step == 1:
        try:
            units = [GovernanceUnit.from_record(row) for row in units_df.to_dict("records")]
            errors.extend(validate_units(units))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"治理单元数据无法解析：{exc}")
        if units_df.empty:
            errors.append("至少需要一个治理单元。")
    elif step == 2:
        try:
            measures = [Measure.from_record(row) for row in measures_df.to_dict("records")]
            errors.extend(validate_measures(measures))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"措施库无法解析：{exc}")
        if measures_df.empty:
            errors.append("措施库不能为空。")
    elif step in {4, 5, 6, 7} and not result_valid:
        errors.append("尚无有效计算结果，请先运行优化计算。")
    return not errors, errors


def next_step(current: int, is_valid: bool, total: int = len(GUIDED_STEPS)) -> int:
    """状态机只在本步校验通过时前进，rerun后值保持。"""
    if not is_valid:
        return current
    return min(current + 1, total - 1)

