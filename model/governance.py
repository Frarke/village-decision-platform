"""报告入库规则、连续重点斑块和治理单元属性提取。空间细则为V1暂定。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, sqrt

import numpy as np
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from .data_adapter import adapt_spatial, parse_crs, polygon_geometry, reproject


@dataclass(frozen=True)
class SpatialConfig:
    risk_threshold: int = 3  # 科技报告表5-1，保留正式决策级默认值
    receptor_distance_m: float = 100.0
    min_area_m2: float = 100.0  # TODO: V1暂定，现场尺度标定
    merge_distance_m: float = 0.0  # TODO: V1暂定，默认只合并共边面
    max_unit_area_m2: float | None = None  # TODO: 可选网格拆分，默认关闭
    max_units: int = 64  # 同步服务资源限制

    def __post_init__(self):
        if type(self.risk_threshold) is not int or not 1 <= self.risk_threshold <= 5:
            raise ValueError("risk_threshold须为1～5整数。")
        for name in ("receptor_distance_m", "min_area_m2", "merge_distance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name}须为有限非负数。")
        if self.max_unit_area_m2 is not None and (not np.isfinite(self.max_unit_area_m2) or self.max_unit_area_m2 <= 0):
            raise ValueError("max_unit_area_m2须大于零或为null。")
        if type(self.max_units) is not int or not 1 <= self.max_units <= 64:
            raise ValueError("max_units须为1～64整数。")


def priority_reasons(props: dict, config: SpatialConfig) -> list[str]:
    reasons = []
    if props["risk"] >= config.risk_threshold:
        reasons.append(f"风险等级≥{config.risk_threshold}")
    if props["distance_m"] <= config.receptor_distance_m:
        reasons.append(f"距承灾体≤{config.receptor_distance_m:g}米")
    if props.get("identified_or_risk_area"):
        reasons.append("已有隐患或风险区识别成果")
    if props.get("focus_unit"):
        reasons.append("已有重点关注标记")
    return reasons


def _components(records, config):
    geoms = [r["geometry"] for r in records]
    tree = STRtree(geoms)
    parents = list(range(len(geoms)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, geom in enumerate(geoms):
        query = geom.buffer(config.merge_distance_m) if config.merge_distance_m else geom
        for j in tree.query(query):
            j = int(j)
            if j <= i or records[i]["properties"]["profile_id"] != records[j]["properties"]["profile_id"]:
                continue
            connected = geom.intersection(geoms[j]).length > 1e-6
            if config.merge_distance_m > 0:
                connected |= geom.distance(geoms[j]) <= config.merge_distance_m
            if connected:
                parents[find(j)] = find(i)
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return [[records[i] for i in indices] for indices in groups.values()]


def _parts(geom):
    if geom.geom_type == "Polygon":
        # 与多面分支采用相同数值容差，排除裁剪边界上的浮点零面积碎片。
        return [geom] if geom.area > 1e-6 else []
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and g.area > 1e-6]


def _split(geom, maximum):
    if maximum is None or geom.area <= maximum:
        return [geom]
    edge = sqrt(maximum)
    x0, y0, x1, y1 = geom.bounds
    nx, ny = ceil((x1 - x0) / edge), ceil((y1 - y0) / edge)
    if nx * ny > 10_000:
        raise ValueError("拆分网格超过上限，请提高单元面积上限。")
    result = []
    for ix in range(nx):
        for iy in range(ny):
            clipped = geom.intersection(box(x0 + ix * edge, y0 + iy * edge,
                                           x0 + (ix + 1) * edge, y0 + (iy + 1) * edge))
            result.extend(_parts(clipped) if not clipped.is_empty else [])
    return result


def build_governance(village: dict, spatial_layers: dict, config: SpatialConfig | None = None) -> dict:
    config = config or SpatialConfig()
    boundary_crs = parse_crs(village.get("crs"))
    boundary = polygon_geometry(village.get("boundary"), "村域边界")
    wgs84 = CRS.from_epsg(4326)
    boundary_wgs = reproject(boundary, boundary_crs, wgs84)
    lon, lat = boundary_wgs.centroid.coords[0]
    bounds = boundary_wgs.bounds
    if not -180 <= lon <= 180 or not -80 <= lat <= 84 or bounds[2] - bounds[0] > 2 or bounds[3] - bounds[1] > 2:
        raise ValueError("V1仅处理有效的村域局部范围（经纬跨度≤2°，纬度-80～84°）。")
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    metric_crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    boundary_metric = reproject(boundary_wgs, wgs84, metric_crs)
    records, source_crs, metadata = adapt_spatial(spatial_layers)
    clipped, outside = [], 0
    for record in records:
        geom = reproject(record["geometry"], source_crs, metric_crs).intersection(boundary_metric)
        polygons = _parts(geom) if not geom.is_empty else []
        if not polygons:
            outside += 1
            continue
        # 一个输入MultiPolygon也可能含不相连斑块，不能仅因共用feature_id便直接当成单元。
        clipped.extend({**record, "geometry": polygon} for polygon in polygons)
    geoms = [r["geometry"] for r in clipped]
    tree = STRtree(geoms)
    for i, geom in enumerate(geoms):
        for j in tree.query(geom):
            if j > i and geom.intersection(geoms[int(j)]).area > 0.001:
                raise ValueError("评价片区存在面积重叠，须先完成多源图层叠置，避免重复统计。")
    selected = []
    for record in clipped:
        reasons = priority_reasons(record["properties"], config)
        if reasons:
            selected.append({**record, "reasons": reasons})
    units, filtered_area, filtered_count = [], 0.0, 0
    for group in _components(selected, config) if selected else []:
        combined = unary_union([r["geometry"] for r in group])
        for unit_geom in _split(combined, config.max_unit_area_m2):
            if unit_geom.area < config.min_area_m2:
                filtered_area += unit_geom.area
                filtered_count += 1
                continue
            portions = [(r, unit_geom.intersection(r["geometry"]).area) for r in group]
            portions = [(r, a) for r, a in portions if a > 1e-6]
            area = sum(a for _, a in portions)
            props = {
                "profile_id": group[0]["properties"]["profile_id"], "area_m2": unit_geom.area,
                "source_feature_ids": sorted({r["properties"]["feature_id"] for r, _ in portions}),
                "risk": max(r["properties"]["risk"] for r, _ in portions),
                "haz": max(r["properties"]["haz"] for r, _ in portions),
                "distance_m": min(r["properties"]["distance_m"] for r, _ in portions),
                "selection_reasons": sorted({reason for r, _ in portions for reason in r["reasons"]}),
            }
            for name in ("slope_deg", "susceptibility", "vulnerability"):
                if all(name in r["properties"] for r, _ in portions):
                    props[f"mean_{name}"] = sum(r["properties"][name] * a for r, a in portions) / area
                    props[f"max_{name}"] = max(r["properties"][name] for r, _ in portions)
            props["slope_deg"] = props["max_slope_deg"]
            geographic = reproject(unit_geom, metric_crs, wgs84)
            props["longitude"], props["latitude"] = reproject(unit_geom.centroid, metric_crs, wgs84).coords[0]
            units.append({"type": "Feature", "geometry": mapping(geographic), "properties": props})
    units.sort(key=lambda f: (f["properties"]["profile_id"], f["properties"]["longitude"], f["properties"]["latitude"]))
    if len(units) > config.max_units:
        raise ValueError(f"生成{len(units)}个单元，超过同步求解上限{config.max_units}；请检查分辨率/分区。")
    for index, feature in enumerate(units, 1):
        feature["properties"]["unit_id"] = f"U{index:02d}"
    status = village.get("data_status", "UNSPECIFIED")
    return {
        "evaluation_areas": {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(reproject(r["geometry"], metric_crs, wgs84)),
             "properties": r["properties"]} for r in clipped], "data_status": status},
        "governance_units": {"type": "FeatureCollection", "features": units, "data_status": status},
        "priority_areas": {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(reproject(r["geometry"], metric_crs, wgs84)),
             "properties": {**r["properties"], "selection_reasons": r["reasons"]}} for r in selected],
            "data_status": status},
        "boundary": mapping(boundary_wgs),
        "spatial_metrics": {**metadata, "metric_crs": metric_crs.to_string(),
            "clipped_features": len(clipped), "outside_features": outside,
            "priority_features": len(selected), "priority_area_m2": sum(r["geometry"].area for r in selected),
            "governance_unit_count": len(units), "filtered_patch_count": filtered_count,
            "filtered_area_m2": filtered_area, "configuration": asdict(config),
            "covered_village_area_m2": sum(g.area for g in geoms), "village_area_m2": boundary_metric.area},
    }
