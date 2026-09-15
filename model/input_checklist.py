"""无服务器草稿状态的逐文件入口；每次校验和计算使用调用方提交的完整文件集。"""
import base64
import binascii
import csv
import json
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from .data_adapter import MAX_FILE_BYTES, geotiff_grid, polygon_geometry, parse_crs, _number
from .input_package import MAX_PACKAGE_BYTES, read_input_package
from .schemas import BOOL_COLUMNS_UNITS
from .service import PROFILE_FIELDS, REQUIRED_PROFILE, _request_config, model_context, prepare_request

CATALOG = [
    ("project.json", "项目与计算参数", True, "JSON；项目来源、数据性质、预算和求解设置"),
    ("boundary.geojson", "研究范围", True, "WGS84 单个有效面；限定计算范围"),
    ("risk.tif", "已有风险评价", True, "单波段 GeoTIFF；1–5 整数等级，用于重点区域识别"),
    ("haz.tif", "已有危险性评价", True, "单波段 GeoTIFF；1–5 整数等级，用于工程适用性判断"),
    ("slope_deg.tif", "地形坡度", True, "单波段 GeoTIFF；0–90 度"),
    ("distance_m.tif", "距承灾体距离", True, "单波段 GeoTIFF；非负米数"),
    ("profile_id.tif", "工程调查分区编码", True, "单波段 GeoTIFF；编码对应下方调查表"),
    ("engineering_profiles.csv", "工程条件调查表", True, "UTF-8 CSV；稳定性、坡高、排水与施工条件，按 profile_id 关联"),
    ("decision_context.geojson", "空间风险权重与路径", True, "WGS84 点与三点折线；显式权重、物源—输移—承灾路径"),
    ("susceptibility.tif", "已有易发性评价", False, "单波段 GeoTIFF；0–1，用于解释与统计"),
    ("vulnerability.tif", "已有易损性评价", False, "单波段 GeoTIFF；0–1，用于解释与统计"),
]
NAMES = {row[0] for row in CATALOG}


def template_files():
    path = Path(__file__).resolve().parents[1] / "data/v1_1/peitian_input_package.zip"
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in NAMES}


def encode_files(files):
    return {name: {"name": name, "content": base64.b64encode(data).decode("ascii")} for name, data in files.items()}


def decode_files(records):
    if not isinstance(records, dict) or set(records) - NAMES:
        raise ValueError("files须按资料清单文件名组织，不支持清单外文件。")
    files, errors, total = {}, {}, 0
    for name, record in records.items():
        if record is None:
            continue
        try:
            if not isinstance(record, dict) or not isinstance(record.get("name"), str):
                raise ValueError("缺少上传文件名。")
            if Path(record["name"]).suffix.lower() != Path(name).suffix:
                raise ValueError(f"文件扩展名须为 {Path(name).suffix}。")
            data = base64.b64decode(record.get("content", ""), validate=True)
            if not data or len(data) > MAX_FILE_BYTES:
                raise ValueError("文件为空或超过20MB。")
            total += len(data)
            files[name] = data
        except (ValueError, TypeError, binascii.Error) as exc:
            errors[name] = str(exc)
    if total > MAX_PACKAGE_BYTES:
        raise ValueError("资料总量超过50MB。")
    return files, errors


def _json(data):
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise ValueError("文件须使用UTF-8编码。") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON格式错误：第{exc.lineno}行、第{exc.colno}列，请检查括号、引号与逗号。") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON根节点须为对象。")
    return value


def _check_file(name, data):
    if name.endswith(".tif"):
        field = name[:-4]
        grid = geotiff_grid(data, [field])
        values = np.asarray(grid["layers"][field], dtype=float)
        valid = values[np.isfinite(values)]
        if not len(valid):
            raise ValueError("没有有效像元。")
        low, high = {"risk": (1, 5), "haz": (1, 5), "slope_deg": (0, 90),
                     "susceptibility": (0, 1), "vulnerability": (0, 1)}.get(field, (0, np.inf))
        if np.any(valid < low) or np.any(valid > high):
            raise ValueError(f"{field}数值须在{low}至{high}范围内。")
        if field in {"risk", "haz", "profile_id"} and np.any(valid != np.floor(valid)):
            raise ValueError("等级或分区编码须为整数。")
        return (parse_crs(grid["crs"]), grid["transform"], values.shape)
    if name == "engineering_profiles.csv":
        table = csv.DictReader(StringIO(data.decode("utf-8-sig")))
        fields = table.fieldnames or []
        if len(fields) != len(set(fields)) or not REQUIRED_PROFILE | {"profile_id"} <= set(fields):
            raise ValueError("调查表列名重复或缺少必填列，请按内置模板补齐。")
        if set(fields) - PROFILE_FIELDS - {"profile_id"}:
            raise ValueError("调查表存在模板以外的列。")
        ids = set()
        for row in table:
            pid = (row.get("profile_id") or "").strip()
            if not pid or pid in ids or None in row or any(v is None for v in row.values()):
                raise ValueError("调查表编号须非空唯一，每行列数须与表头相同。")
            ids.add(pid)
            for field in REQUIRED_PROFILE:
                value = row[field]
                if field in BOOL_COLUMNS_UNITS:
                    if value.lower() not in {"true", "false"}:
                        raise ValueError(f"{pid}.{field}须为true或false。")
                else:
                    try:
                        numeric = float(value)
                    except ValueError as exc:
                        raise ValueError(f"{pid}.{field}须为数值。") from exc
                    _number(numeric, f"{pid}.{field}", 0, integer=field == "gully_level")
        if not ids:
            raise ValueError("调查表没有记录。")
        return None
    value = _json(data)
    if name == "project.json":
        if not {"schema_version", "village", "raster_files", "configuration"} <= set(value):
            raise ValueError("缺少项目必填字段，请参照模板。")
        _request_config({k: value[k] for k in ("schema_version", "village", "configuration")})
        raster_files = value["raster_files"]
        if not isinstance(raster_files, dict) or any(f"{k}.tif" != v or v not in NAMES for k, v in raster_files.items()):
            raise ValueError("逐项入口的raster_files须使用清单中的标准文件名。")
    elif name == "boundary.geojson":
        if value.get("type") == "FeatureCollection" and len(value.get("features", [])) == 1:
            value = value["features"][0]
        if value.get("type") != "Feature":
            raise ValueError("研究范围须为单个面Feature。")
        geom = polygon_geometry(value.get("geometry"), "研究范围")
        x0, y0, x1, y1 = geom.bounds
        if not (-180 <= x0 <= x1 <= 180 and -90 <= y0 <= y1 <= 90):
            raise ValueError("研究范围须使用WGS84经纬度。")
    elif value.get("type") != "FeatureCollection" or not isinstance(value.get("features"), list):
        raise ValueError("空间权重与路径须为FeatureCollection。")
    return None


def validate_files(files, upload_errors=None, total_budget=None):
    """返回逐项格式状态、跨文件校验及可计算请求；不回退至默认文件。"""
    errors = dict(upload_errors or {})
    if sum(len(data) for data in files.values()) > MAX_PACKAGE_BYTES:
        errors["project.json"] = "资料总量超过50MB。"
    signatures = {}
    for name, data in files.items():
        try:
            if not data or len(data) > MAX_FILE_BYTES:
                raise ValueError("文件为空或超过20MB。")
            signature = _check_file(name, data)
            if signature is not None:
                signatures[name] = signature
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
            errors[name] = str(exc)
    if signatures and any(s != next(iter(signatures.values())) for s in signatures.values()):
        for name in signatures:
            errors[name] = "栅格组未对齐：CRS、像元大小、行列数或范围不一致。"
    rows = [{"key": name, "title": title, "required": required, "description": desc,
             "present": name in files or name in errors,
             "valid": name in files and name not in errors,
             "message": errors.get(name, "格式通过" if name in files else "缺少必需文件" if required else "未提供（可选）")}
            for name, title, required, desc in CATALOG]
    report = {"items": rows, "valid": False, "errors": [], "summary": None}
    if errors or any(r["required"] and not r["valid"] for r in rows):
        report["errors"] = ["请修正红色资料项；未通过校验的替换文件不会使用内置文件代替。"]
        return report, None
    related = {"project.json"}
    try:
        # 可选图层以清单中的实际存在情况为准，禁止残留清单引用。
        manifest = _json(files["project.json"])
        for field in ("susceptibility", "vulnerability"):
            if f"{field}.tif" in files:
                manifest["raster_files"][field] = f"{field}.tif"
            else:
                manifest["raster_files"].pop(field, None)
        buffer = BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(name, json.dumps(manifest, ensure_ascii=False) if name == "project.json" else content)
        request = read_input_package(buffer.getvalue())
        if total_budget is not None:
            request["configuration"].setdefault("solver", {})["total_budget"] = _number(total_budget, "总预算", 0)
        if request["village"]["data_status"] == "REAL":
            defaults = template_files()
            if any(data == defaults[name] for name, data in files.items() if name != "project.json"):
                raise ValueError("仍含内置模拟文件，项目不能声明为REAL；请替换为同一研究区的真实资料。")
        related = {"boundary.geojson", "decision_context.geojson"} | set(signatures)
        prepared = prepare_request(request)
        if prepared["status"] == "no_governance_units":
            raise ValueError("当前资料与规则未识别出治理单元，请复核研究范围及评价数据。")
        related = {"engineering_profiles.csv", "decision_context.geojson", "project.json"}
        model_context(request, prepared)  # 包含工程字段、权重落区、路径、约束的完整校验。
        report.update(valid=True, summary={"village": request["village"]["name"],
                      "data_status": request["village"]["data_status"], "input_hash": prepared["input_hash"],
                      "governance_units": prepared["spatial_metrics"]["governance_unit_count"],
                      "budget": request["configuration"].get("solver", {}).get("total_budget")})
        return report, request
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        report["errors"] = ["资料关联校验未通过：" + str(exc)]
        for row in rows:
            if row["key"] in related:
                row.update(valid=False, message="格式通过；关联校验待修正，详见上方校验说明。")
        return report, None
