"""治理单元、措施库、GeoJSON与空间拓扑的一致性校验。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .schemas import GovernanceUnit, Measure, validate_measures, validate_units


def validate_project_data(
    units_df: pd.DataFrame,
    measures_df: pd.DataFrame,
    units_geojson: dict[str, Any],
    topology: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        units = [GovernanceUnit.from_record(row) for row in units_df.to_dict("records")]
        errors.extend(validate_units(units))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"治理单元CSV字段或类型错误：{exc}")
        units = []
    try:
        measures = [Measure.from_record(row) for row in measures_df.to_dict("records")]
        errors.extend(validate_measures(measures))
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"措施库字段或类型错误：{exc}")
        measures = []
    table_ids = {unit.unit_id for unit in units}
    geo_ids = [str(feature.get("properties", {}).get("unit_id", "")) for feature in units_geojson.get("features", [])]
    if any(not value for value in geo_ids):
        errors.append("GeoJSON存在缺少unit_id的治理单元。")
    if len(geo_ids) != len(set(geo_ids)):
        errors.append("GeoJSON治理单元编号必须唯一。")
    if table_ids != set(geo_ids):
        errors.append("治理单元CSV与GeoJSON的unit_id集合不一致。")
    for path in topology.get("paths", []):
        for key, label in (("source_unit", "物源"), ("transport_unit", "输移"), ("receptor_unit", "承灾体")):
            if str(path.get(key, "")) not in table_ids:
                errors.append(f"风险路径{path.get('path_id', '未编号')}的{label}单元不存在。")
    if not measures:
        errors.append("措施库不能为空。")
    return list(dict.fromkeys(errors))

