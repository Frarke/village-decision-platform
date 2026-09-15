"""文件格式适配到基础评价面；不包含工程评价或措施规则。"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from affine import Affine
from pyproj import CRS, Transformer
from shapely import from_wkt
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform


MAX_CELLS = 10_000  # 同步 V1 服务的资源上限，不是科研阈值。
MAX_FILE_BYTES = 20_000_000
REQUIRED_SPATIAL = ("risk", "haz", "distance_m", "slope_deg", "profile_id")


def polygon_geometry(value: dict, label: str):
    if not isinstance(value, dict):
        raise ValueError(f"{label}须提供geometry对象。")
    try:
        geom = shape(value)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
        raise ValueError(f"{label}几何无法解析：{exc}") from exc
    if geom.geom_type not in {"Polygon", "MultiPolygon"} or geom.is_empty or not geom.is_valid:
        raise ValueError(f"{label}须为非空且有效的 Polygon/MultiPolygon，不能用图片或点代替。")
    if not np.isfinite(geom.bounds).all():
        raise ValueError(f"{label}坐标必须是有限数值。")
    return geom


def parse_crs(value: Any) -> CRS:
    if not value:
        raise ValueError("空间数据必须显式提供 CRS。")
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f"无法解析 CRS：{value}") from exc


def reproject(geom, source: CRS, target: CRS):
    if source == target:
        return geom
    converted = transform(Transformer.from_crs(source, target, always_xy=True).transform, geom)
    if converted.is_empty or not converted.is_valid or not np.isfinite(converted.bounds).all():
        raise ValueError("坐标转换失败，请核对 CRS 和坐标范围。")
    return converted


def _number(value, name, low, high=None, integer=False):
    if isinstance(value, (bool, str)) or value is None:
        raise ValueError(f"{name}必须提供数值。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须提供数值。") from exc
    if not np.isfinite(number) or number < low or (high is not None and number > high):
        raise ValueError(f"{name}超出范围或不是有限数值。")
    if integer and number != int(number):
        raise ValueError(f"{name}必须为整数等级。")
    return int(number) if integer else number


def validate_properties(props: dict, label: str) -> dict:
    missing = [key for key in REQUIRED_SPATIAL if key not in props]
    if missing:
        raise ValueError(f"{label}缺少空间字段：{', '.join(missing)}")
    result = dict(props)
    for key in ("risk", "haz"):
        result[key] = _number(props[key], f"{label}.{key}", 1, 5, integer=True)
    result["distance_m"] = _number(props["distance_m"], f"{label}.distance_m", 0)
    result["slope_deg"] = _number(props["slope_deg"], f"{label}.slope_deg", 0, 90)
    for key in ("susceptibility", "vulnerability"):
        if key in props:
            result[key] = _number(props[key], f"{label}.{key}", 0, 1)
    for key in ("identified_or_risk_area", "focus_unit"):
        value = props.get(key, False)
        if value not in (True, False, 0, 1) or isinstance(value, str):
            raise ValueError(f"{label}.{key}必须是布尔值。")
        result[key] = bool(value)
    if isinstance(props["profile_id"], float) and props["profile_id"].is_integer():
        result["profile_id"] = str(int(props["profile_id"]))
    else:
        result["profile_id"] = str(props["profile_id"]).strip()
    if not result["profile_id"] or props["profile_id"] is None:
        raise ValueError(f"{label}.profile_id不能为空。")
    return result


def grid_features(spec: dict) -> tuple[list[dict], dict]:
    layers = spec.get("layers", {})
    if not isinstance(layers, dict) or not layers:
        raise ValueError("栅格输入缺少 layers。")
    missing = set(REQUIRED_SPATIAL) - set(layers)
    if missing:
        raise ValueError(f"栅格缺少波段字段：{', '.join(sorted(missing))}")
    try:
        arrays = {name: np.asarray(values, dtype=object) for name, values in layers.items()}
    except ValueError as exc:
        raise ValueError("栅格数组不是规则的二维网格。") from exc
    size = next(iter(arrays.values())).shape
    if len(size) != 2 or not all(size) or any(a.shape != size for a in arrays.values()):
        raise ValueError("栅格各波段必须是同一行列数的二维网格，不能静默重采样。")
    if size[0] * size[1] > MAX_CELLS:
        raise ValueError(f"栅格超过同步处理上限 {MAX_CELLS} 个像元，请裁剪或合理降采样。")
    values = spec.get("transform", [])
    if len(values) != 6 or not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("transform须为有限的[a,b,c,d,e,f]六参数。")
    affine = Affine(*values)
    if abs(affine.determinant) < 1e-12:
        raise ValueError("栅格仿射变换不可逆。")
    nodata = spec.get("nodata")
    features = []
    excluded = 0
    for row in range(size[0]):
        for col in range(size[1]):
            props = {name: array[row, col] for name, array in arrays.items()}
            if any(v is None or (isinstance(v, (int, float)) and not np.isfinite(v)) or
                   (nodata is not None and v == nodata) for v in props.values()):
                excluded += 1
                continue
            def xy(c, r):
                return affine.a * c + affine.b * r + affine.c, affine.d * c + affine.e * r + affine.f
            geom = Polygon([xy(col, row), xy(col + 1, row), xy(col + 1, row + 1), xy(col, row + 1)])
            features.append({"type": "Feature", "geometry": mapping(geom),
                             "properties": {"feature_id": f"R{row:05d}C{col:05d}", **props}})
    return features, {"raster_shape": list(size), "nodata_cells": excluded, "transform": list(values)}


def geotiff_grid(content: bytes, band_fields: list[str]) -> dict:
    import rasterio
    from rasterio.io import MemoryFile
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("GeoTIFF超过20MB输入上限。")
    try:
        with MemoryFile(content) as memory, memory.open() as source:
            if source.crs is None:
                raise ValueError("GeoTIFF没有 CRS。")
            if source.width * source.height > MAX_CELLS:
                raise ValueError(f"GeoTIFF超过 {MAX_CELLS} 像元，请先裁剪研究区。")
            if len(band_fields) != source.count or len(set(band_fields)) != len(band_fields):
                raise ValueError("band_fields须与波段数量一致且字段唯一。")
            layers = {}
            for index, field in enumerate(band_fields, 1):
                array = source.read(index, masked=True).astype(float)
                values = array.filled(np.nan).astype(object)
                values[~np.isfinite(array.filled(np.nan))] = None
                layers[field] = values.tolist()
            return {"kind": "raster_grid", "crs": source.crs.to_string(),
                    "transform": list(source.transform)[:6], "layers": layers}
    except rasterio.errors.RasterioError as exc:
        raise ValueError(f"GeoTIFF无法读取：{exc}") from exc


def adapt_spatial(spec: dict) -> tuple[list[dict], CRS, dict]:
    if not isinstance(spec, dict):
        raise ValueError("spatial_layers须为带kind的对象。")
    kind = spec.get("kind")
    metadata = {"adapter": kind}
    if kind == "geotiff":
        try:
            content = base64.b64decode(spec["content_base64"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("GeoTIFF须提供有效的content_base64。") from exc
        spec = geotiff_grid(content, spec.get("band_fields", []))
        kind = "raster_grid"
    crs = parse_crs(spec.get("crs"))
    if kind == "vector":
        features = spec.get("features")
    elif kind == "raster_grid":
        features, grid_meta = grid_features(spec)
        metadata.update(grid_meta)
    else:
        raise ValueError("不支持的空间输入；请提供vector、raster_grid或geotiff，PNG/JPG不是GIS计算数据。")
    if not isinstance(features, list) or len(features) > MAX_CELLS:
        raise ValueError(f"features须为列表且不超过 {MAX_CELLS} 个评价片区。")
    output, ids = [], set()
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
            raise ValueError("每个评价片区须为包含properties的Feature。")
        fid = str(feature["properties"].get("feature_id", "")).strip()
        if not fid or fid in ids:
            raise ValueError("评价片区 feature_id 必须非空且唯一。")
        ids.add(fid)
        props = validate_properties(feature["properties"], fid)
        props["feature_id"] = fid
        geom = polygon_geometry(feature.get("geometry"), fid)
        if crs.is_geographic and (geom.bounds[0] < -180 or geom.bounds[2] > 180 or
                                  geom.bounds[1] < -90 or geom.bounds[3] > 90):
            raise ValueError(f"{fid}经纬度越界，请核对CRS。")
        output.append({"geometry": geom, "properties": props})
    for key in ("susceptibility", "vulnerability"):
        if any(key in f["properties"] for f in output) and not all(key in f["properties"] for f in output):
            raise ValueError(f"{key}图层不完整；须覆盖全部有效评价片区或整体省略。")
    metadata.update({"source_crs": crs.to_string(), "valid_features": len(output)})
    return output, crs, metadata


def read_spatial_file(path: str | Path, *, crs: str | None = None, band_fields=None) -> dict:
    """本地调用适配器；HTTP入口只接受文件内容，不接收服务器文件路径。"""
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("空间文件超过20MB输入上限。")
    if path.suffix.lower() in {".tif", ".tiff"}:
        return geotiff_grid(path.read_bytes(), band_fields or [])
    if path.suffix.lower() in {".geojson", ".json"}:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("type") == "FeatureCollection":
            return {"kind": "vector", "crs": crs or "EPSG:4326", "features": data["features"]}
        return data
    if path.suffix.lower() in {".csv", ".xlsx"}:
        frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
        if "geometry_wkt" not in frame:
            raise ValueError("空间属性表必须有geometry_wkt几何列。")
        parse_crs(crs)
        features = []
        for row in frame.to_dict("records"):
            geom = from_wkt(row.pop("geometry_wkt"))
            features.append({"type": "Feature", "geometry": mapping(geom), "properties": row})
        return {"kind": "vector", "crs": crs, "features": features}
    raise ValueError("该格式尚未接入；支持GeoJSON、GeoTIFF、CSV/WKT和XLSX/WKT。普通图片不能用于计算。")
