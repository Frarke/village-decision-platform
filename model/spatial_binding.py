"""将显式权重及路径坐标绑定到生成单元；不生成风险权重或推断风险路径。"""
from __future__ import annotations

from shapely.geometry import Point, shape

from .data_adapter import _number


def bind_spatial_context(context: dict, units: dict) -> tuple[dict, dict, list]:
    if not isinstance(context, dict) or context.get("type") != "FeatureCollection":
        raise ValueError("spatial_bindings须为WGS84 FeatureCollection。")
    if not context.get("source") or context.get("data_status") not in {"REAL", "MOCK"}:
        raise ValueError("空间权重与路径须声明source及REAL/MOCK数据性质。")
    features = context.get("features")
    if not isinstance(features, list) or len(features) > 512:
        raise ValueError("空间权重及路径记录须为列表且不超过512条。")
    geometries = [(f["properties"]["unit_id"], shape(f["geometry"])) for f in units["features"]]

    def locate(coordinate, label):
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError(f"{label}须为[经度,纬度]。")
        lon = _number(coordinate[0], label + "经度", -180, 180)
        lat = _number(coordinate[1], label + "纬度", -90, 90)
        matches = [uid for uid, geom in geometries if geom.covers(Point(lon, lat))]
        if len(matches) != 1:
            raise ValueError(f"{label}对应{len(matches)}个治理单元；请复核坐标/划分，程序不就近猜测。")
        return matches[0]

    weights, paths, trace = {}, [], []
    seen = set()
    for feature in features:
        if not isinstance(feature, dict):
            raise ValueError("空间绑定记录须为Feature。")
        props, geom = feature.get("properties", {}), feature.get("geometry", {})
        fid = str(props.get("record_id", "")).strip()
        if not fid or fid in seen:
            raise ValueError("空间绑定record_id须非空且唯一。")
        seen.add(fid)
        if props.get("kind") == "weight" and geom.get("type") == "Point":
            uid = locate(geom.get("coordinates"), fid)
            if uid in weights:
                raise ValueError(f"{uid}包含多个显式权重点，须复核而非自动平均。")
            weights[uid] = _number(props.get("risk_weight"), fid + ".risk_weight", 0)
            trace.append({"record_id": fid, "unit_id": uid, "risk_weight": weights[uid], "method": "点落入单元，原权重数值不变"})
        elif props.get("kind") == "path" and geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            if len(coords) != 3 or type(props.get("key", True)) is not bool:
                raise ValueError(f"{fid}路径须有物源、输移、承灾三个坐标点，key须为布尔值。")
            path = {"path_id": fid, "name": str(props.get("name", fid)), "key": props.get("key", True)}
            for stage, coordinate in zip(("source", "transport", "receptor"), coords, strict=True):
                path[stage + "_unit"] = locate(coordinate, fid + "." + stage)
            paths.append(path)
        else:
            raise ValueError(f"{fid}仅支持weight点或path三点折线。")
    missing = {uid for uid, _ in geometries} - set(weights)
    if missing:
        raise ValueError("以下生成单元缺少显式空间权重点：" + "、".join(sorted(missing)))
    if not paths and context.get("status") != "confirmed_empty":
        raise ValueError("未提供空间路径；确认无路径时须显式设置status=confirmed_empty。")
    topology = {"source": context["source"], "data_status": context["data_status"], "paths": paths}
    if not paths:
        topology["status"] = "confirmed_empty"
    return weights, topology, trace
