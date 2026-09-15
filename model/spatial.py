"""示例GeoJSON与风险传播拓扑的加载、校验和协同评价。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import shape


STAGE_CONTROL_MEASURES = {
    "source": {"截排水工程", "植被混凝土", "格构锚固", "挡墙工程", "抗滑桩"},
    "transport": {"截排水工程", "谷坊坝", "柔性拦挡结构"},
    "receptor": {"挡墙工程", "抗滑桩", "柔性拦挡结构"},
}
PATH_CONTROL_MEASURES = set().union(*STAGE_CONTROL_MEASURES.values())


def _detail_measure_names(detail: dict[str, object]) -> set[str]:
    names = detail.get("measure_names")
    if isinstance(names, list):
        return {str(item) for item in names if str(item)}
    raw = str(detail.get("measures", "不治理"))
    return {item for item in raw.split("＋") if item and item != "不治理"}


def load_geojson(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"空间数据无法加载：{Path(path).name}；{exc}") from exc
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError(f"{Path(path).name}不是有效的GeoJSON要素集合。")
    return data


def load_topology(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"空间拓扑无法加载：{Path(path).name}；{exc}") from exc
    if not isinstance(data.get("paths"), list):
        raise ValueError("空间拓扑缺少paths列表。")
    return data


def polygon_centroid(feature: dict[str, Any]) -> tuple[float, float]:
    point = shape(feature["geometry"]).representative_point()
    return point.x, point.y


def validate_topology(topology: dict, unit_ids: set[str]) -> None:
    if not isinstance(topology, dict) or not isinstance(topology.get("paths"), list):
        raise ValueError("topology须包含paths列表；缺少路径不能默认为无风险路径。")
    ids = set()
    for path in topology["paths"]:
        if not isinstance(path, dict):
            raise ValueError("每条风险路径须为对象。")
        pid = path.get("path_id")
        if not isinstance(pid, str) or not pid.strip() or pid in ids or not path.get("name"):
            raise ValueError("风险路径须有唯一path_id和非空name。")
        ids.add(pid)
        if "key" in path and type(path["key"]) is not bool:
            raise ValueError("路径key标记须为布尔值。")
        for field in ("source_unit", "transport_unit", "receptor_unit"):
            if path.get(field) not in unit_ids:
                raise ValueError(f"路径{pid}的{field}引用不存在的治理单元。")


def spatial_unit_table(units_geojson: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feature in units_geojson["features"]:
        lon, lat = polygon_centroid(feature)
        rows.append({**feature.get("properties", {}), "longitude": lon, "latitude": lat})
    return pd.DataFrame(rows)


def path_stage_flags(detail_map: dict[str, dict[str, object]], path: dict[str, object]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for stage_name, unit_key in (
        ("source", "source_unit"),
        ("transport", "transport_unit"),
        ("receptor", "receptor_unit"),
    ):
        unit_id = str(path[unit_key])
        measures = _detail_measure_names(detail_map.get(unit_id, {}))
        flags[stage_name] = bool(measures & STAGE_CONTROL_MEASURES[stage_name])
    return flags


def evaluate_path_coverage(
    topology: dict[str, Any],
    plan_details: list[dict[str, object]],
) -> dict[str, object]:
    detail_map = {str(item["unit_id"]): item for item in plan_details}
    paths = [p for p in topology.get("paths", []) if p.get("key", True)]
    path_rows: list[dict[str, object]] = []
    covered = partial = 0
    source_hits = transport_hits = receptor_hits = 0
    residuals: list[float] = []
    weak_links: list[str] = []
    for path in paths:
        flags = path_stage_flags(detail_map, path)
        status = [flags["source"], flags["transport"], flags["receptor"]]
        source_hits += int(flags["source"])
        transport_hits += int(flags["transport"])
        receptor_hits += int(flags["receptor"])
        residuals.append(1.0 - sum(status) / 3.0)
        if all(status):
            covered += 1
            summary = "完整控制"
        elif any(status):
            partial += 1
            summary = "部分控制"
            weak_links.append(f"{path['name']}存在未覆盖环节")
        else:
            summary = "未控制"
            weak_links.append(f"{path['name']}尚未形成工程防线")
        path_rows.append(
            {
                "path_id": str(path.get("path_id", "")),
                "name": str(path.get("name", "")),
                "source_unit": str(path.get("source_unit", "")),
                "transport_unit": str(path.get("transport_unit", "")),
                "receptor_unit": str(path.get("receptor_unit", "")),
                "source_controlled": flags["source"],
                "transport_controlled": flags["transport"],
                "receptor_controlled": flags["receptor"],
                "fully_covered": all(status),
                "partially_covered": any(status) and not all(status),
                "status_summary": summary,
            }
        )
    total = len(paths)
    coverage = 100.0 if total == 0 else 100.0 * covered / total
    source_rate = 100.0 if total == 0 else 100.0 * source_hits / total
    transport_rate = 100.0 if total == 0 else 100.0 * transport_hits / total
    receptor_rate = 100.0 if total == 0 else 100.0 * receptor_hits / total
    return {
        "N_key": total,
        "N_covered": covered,
        "partial_paths": partial,
        "coverage_score": round(coverage, 6),
        "source_rate": round(source_rate, 6),
        "transport_rate": round(transport_rate, 6),
        "receptor_rate": round(receptor_rate, 6),
        "coordination_score": round((source_rate + transport_rate + receptor_rate) / 3.0, 6),
        "average_residual_risk": round(100.0 * sum(residuals) / max(total, 1), 6),
        "weak_links": weak_links,
        "path_details": path_rows,
    }


def coordination_metrics(topology: dict[str, Any], plan_details: list[dict[str, object]]) -> dict[str, object]:
    summary = evaluate_path_coverage(topology, plan_details)
    configurations = [str(item.get("measures", "不治理")) for item in plan_details]
    duplicate_count = len(configurations) - len(set(configurations))
    return {
        "路径总数": int(summary["N_key"]),
        "完整覆盖路径数": int(summary["N_covered"]),
        "部分覆盖路径数": int(summary["partial_paths"]),
        "路径治理覆盖率（%）": round(float(summary["coverage_score"]), 2),
        "物源控制率（%）": round(float(summary["source_rate"]), 2),
        "输移拦截率（%）": round(float(summary["transport_rate"]), 2),
        "承灾体防护率（%）": round(float(summary["receptor_rate"]), 2),
        "上下游协同贡献（%）": round(float(summary["coordination_score"]), 2),
        "路径平均残余风险（%）": round(float(summary["average_residual_risk"]), 2),
        "重复配置单元数": duplicate_count,
        "未治理薄弱环节": "；".join(summary["weak_links"]) if summary["weak_links"] else "无",
        "路径明细": summary["path_details"],
    }
