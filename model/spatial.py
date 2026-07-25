"""示例GeoJSON与风险传播拓扑的加载、校验和协同评价。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


STAGE_CONTROL_MEASURES = {
    "source": {"截排水工程", "植被混凝土", "格构锚固", "挡墙工程", "抗滑桩"},
    "transport": {"截排水工程", "谷坊坝", "柔性拦挡结构"},
    "receptor": {"挡墙工程", "抗滑桩", "柔性拦挡结构"},
}
PATH_CONTROL_MEASURES = set().union(*STAGE_CONTROL_MEASURES.values())


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
    ring = feature["geometry"]["coordinates"][0]
    return sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)


def spatial_unit_table(units_geojson: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feature in units_geojson["features"]:
        lon, lat = polygon_centroid(feature)
        rows.append({**feature.get("properties", {}), "longitude": lon, "latitude": lat})
    return pd.DataFrame(rows)


def coordination_metrics(topology: dict[str, Any], plan_details: list[dict[str, object]]) -> dict[str, object]:
    details = {str(item["unit_id"]): item for item in plan_details}
    paths = topology.get("paths", [])
    covered = 0
    partial = 0
    weak_links: list[str] = []
    source_hits = transport_hits = receptor_hits = 0
    residuals: list[float] = []
    for path in paths:
        ids = [str(path["source_unit"]), str(path["transport_unit"]), str(path["receptor_unit"])]
        stages = []
        for stage_name, unit_id in zip(("source", "transport", "receptor"), ids, strict=True):
            measures = set(str(details.get(unit_id, {}).get("measures", "不治理")).split("＋"))
            stages.append(bool(measures & STAGE_CONTROL_MEASURES[stage_name]))
        source_hits += int(stages[0])
        transport_hits += int(stages[1])
        receptor_hits += int(stages[2])
        residuals.append(1.0 - sum(stages) / 3.0)
        if all(stages):
            covered += 1
        elif any(stages):
            partial += 1
            weak_links.append(f"{path['name']}存在未覆盖环节")
        else:
            weak_links.append(f"{path['name']}尚未形成工程防线")
    total = len(paths)
    coverage = 100.0 if total == 0 else 100.0 * covered / total
    source_rate = 100.0 if not paths else 100.0 * source_hits / total
    transport_rate = 100.0 if not paths else 100.0 * transport_hits / total
    receptor_rate = 100.0 if not paths else 100.0 * receptor_hits / total
    configurations = [str(item.get("measures", "不治理")) for item in plan_details]
    duplicate_count = len(configurations) - len(set(configurations))
    return {
        "路径总数": total,
        "完整覆盖路径数": covered,
        "部分覆盖路径数": partial,
        "路径治理覆盖率（%）": round(coverage, 2),
        "物源控制率（%）": round(source_rate, 2),
        "输移拦截率（%）": round(transport_rate, 2),
        "承灾体防护率（%）": round(receptor_rate, 2),
        "上下游协同贡献（%）": round((source_rate + transport_rate + receptor_rate) / 3.0, 2),
        "路径平均残余风险（%）": round(100.0 * sum(residuals) / max(total, 1), 2),
        "重复配置单元数": duplicate_count,
        "未治理薄弱环节": "；".join(weak_links) if weak_links else "无",
    }
