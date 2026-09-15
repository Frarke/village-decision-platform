"""按业务资料文件组装请求；ZIP仅在内存读取，不解压到服务器文件系统。"""
from __future__ import annotations

import csv
import json
from io import BytesIO, StringIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile
from shapely.geometry import mapping, shape

from .data_adapter import MAX_FILE_BYTES, REQUIRED_SPATIAL, geotiff_grid, parse_crs, reproject
from .schemas import BOOL_COLUMNS_UNITS

MAX_PACKAGE_BYTES = 50_000_000


def read_input_package(content: bytes) -> dict:
    if len(content) > MAX_PACKAGE_BYTES:
        raise ValueError("资料包超过50MB。")
    try:
        archive = ZipFile(BytesIO(content))
    except BadZipFile as exc:
        raise ValueError("资料包不是有效ZIP文件。") from exc
    with archive:
        info = archive.infolist()
        names = [f.filename for f in info]
        if len(names) != len(set(names)) or len(names) > 40:
            raise ValueError("资料包有重复文件名或文件数超过40。")
        if sum(f.file_size for f in info) > MAX_PACKAGE_BYTES or any(f.file_size > MAX_FILE_BYTES for f in info):
            raise ValueError("资料包展开后超过50MB或单文件超过20MB。")
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts or "\\" in n for n in names):
            raise ValueError("资料包中存在不允许的路径。")

        def read(name):
            if not isinstance(name, str) or name not in names:
                raise ValueError(f"资料包缺少文件：{name}")
            try:
                return archive.read(name)
            except (BadZipFile, RuntimeError, NotImplementedError) as exc:
                raise ValueError(f"资料包文件{name}损坏、加密或使用了不支持的压缩格式。") from exc

        def read_json(name):
            try:
                return json.loads(read(name).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{name}不是有效UTF-8 JSON。") from exc

        manifest = read_json("project.json")
        required = {"schema_version", "village", "raster_files", "configuration"}
        if not isinstance(manifest, dict) or not required <= set(manifest):
            raise ValueError("project.json须含schema_version、village、raster_files和configuration。")
        files = manifest["raster_files"]
        if not isinstance(files, dict) or not set(REQUIRED_SPATIAL) <= set(files):
            raise ValueError("栅格清单须含risk、haz、distance_m、slope_deg、profile_id；易发性/易损性可另行提供。")
        if set(files) - set(REQUIRED_SPATIAL) - {"susceptibility", "vulnerability"}:
            raise ValueError("栅格清单包含未支持的字段。")
        layers, reference = {}, None
        for field, name in files.items():
            grid = geotiff_grid(read(name), [field])
            signature = (parse_crs(grid["crs"]), grid["transform"], np.shape(grid["layers"][field]))
            if reference is not None and signature != reference:
                raise ValueError(f"{name}与其他评价栅格的CRS、像元大小或范围不同；请先对齐图层。")
            reference = signature
            layers[field] = grid["layers"][field]
        boundary = read_json("boundary.geojson")
        if boundary.get("type") == "FeatureCollection" and len(boundary.get("features", [])) == 1:
            boundary = boundary["features"][0]
        if boundary.get("type") != "Feature":
            raise ValueError("boundary.geojson须包含一个研究范围面Feature。")
        village = {**manifest["village"], "crs": "EPSG:4326", "boundary": boundary["geometry"]}
        profiles = {}
        for row in csv.DictReader(StringIO(read("engineering_profiles.csv").decode("utf-8-sig"))):
            pid = row.pop("profile_id", "").strip()
            if not pid or pid in profiles:
                raise ValueError("工程调查表profile_id须非空唯一。")
            profile = {}
            for key, value in row.items():
                if value is None:
                    raise ValueError(f"工程调查表{pid}.{key}缺值。")
                if key in BOOL_COLUMNS_UNITS:
                    if value.lower() not in {"true", "false"}:
                        raise ValueError(f"工程调查表{pid}.{key}须为true或false。")
                    profile[key] = value.lower() == "true"
                elif key in {"unit_name", "evidence_level", "notes"}:
                    profile[key] = value
                else:
                    try:
                        profile[key] = int(value) if key == "gully_level" else float(value)
                    except ValueError as exc:
                        raise ValueError(f"工程调查表{pid}.{key}须为数值。") from exc
            profiles[pid] = profile
        return {"schema_version": manifest["schema_version"], "village": village,
                "spatial_layers": {"kind": "raster_grid", "crs": grid["crs"], "transform": grid["transform"], "layers": layers},
                "engineering_profiles": profiles, "spatial_bindings": read_json("decision_context.geojson"),
                "configuration": manifest["configuration"]}


def build_input_package(request: dict) -> bytes:
    """输出可拆开交接的真实GIS资料模板；保留数据性质与来源。"""
    if request["spatial_layers"]["kind"] != "raster_grid" or "spatial_bindings" not in request:
        raise ValueError("文件包导出需要栅格网格与空间权重/路径。")
    spec = request["spatial_layers"]
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        village = {k: v for k, v in request["village"].items() if k not in {"boundary", "crs"}}
        manifest = {"schema_version": request["schema_version"], "village": village,
                    "raster_files": {k: f"{k}.tif" for k in spec["layers"]}, "configuration": request["configuration"]}
        for name, content in {"project.json": manifest, "decision_context.geojson": request["spatial_bindings"],
                              "boundary.geojson": {"type": "Feature", "geometry": mapping(reproject(
                                  shape(request["village"]["boundary"]), parse_crs(request["village"]["crs"]), parse_crs("EPSG:4326"))),
                                                   "properties": {"source": village["source"], "data_status": village["data_status"]}}}.items():
            archive.writestr(name, json.dumps(content, ensure_ascii=False, indent=2))
        for field, values in spec["layers"].items():
            array = np.asarray(values, dtype="float64")
            with MemoryFile() as memory:
                with memory.open(driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                                 dtype="float64", crs=spec["crs"], transform=Affine(*spec["transform"]), nodata=-9999) as ds:
                    ds.write(np.where(np.isfinite(array), array, -9999), 1)
                    ds.set_band_description(1, field)
                    ds.update_tags(data_status=village["data_status"], source=village["source"])
                archive.writestr(field + ".tif", memory.read())
        table = StringIO(newline="")
        fields = ["profile_id"] + list(next(iter(request["engineering_profiles"].values())))
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for pid, profile in request["engineering_profiles"].items():
            writer.writerow({"profile_id": pid, **profile})
        archive.writestr("engineering_profiles.csv", table.getvalue().encode("utf-8-sig"))
        archive.writestr("资料说明.txt", "这些是模型输入资料，不含预设治理单元。单元由评价栅格实际计算。\n"
                         "risk/haz=已有风险/危险性等级；distance_m=距承灾体距离；slope_deg=坡度；profile_id=调查区编码。\n"
                         "susceptibility/vulnerability=已有易发性/易损性；不能代替风险与工程调查。\n"
                         "decision_context.geojson=显式风险权重点及三环节路径坐标，不需要U01编号。\n"
                         "MOCK标记表示模拟；真实地图位置不意味着风险数据真实。")
    return buffer.getvalue()
