"""启动：python -m uvicorn api:app --host 127.0.0.1 --port 8000。"""
import logging
from typing import Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from model.service import optimize_request, prepare_request
from model.input_package import MAX_PACKAGE_BYTES, read_input_package


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    village: dict[str, Any]
    spatial_layers: dict[str, Any]
    engineering_profiles: dict[str, Any] = Field(default_factory=dict)
    unit_risk_weights: dict[str, Any] = Field(default_factory=dict)
    topology: dict[str, Any] | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    measures: list[dict[str, Any]] | None = None
    spatial_bindings: dict[str, Any] | None = None


app = FastAPI(title="村庄地质灾害综合整治决策服务", version="1.1.0",
              description="空间评价成果进入，返回自动治理单元、非劣方案与空间治理结果。")
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return {"status": "ok", "schema_version": "1.0"}


async def invoke(function, request):
    try:
        return await run_in_threadpool(function, request.model_dump(exclude_none=True))
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("输入或方案校验失败：%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/prepare")
async def prepare(request: DecisionRequest):
    """解析空间数据并生成单元，返回需显式赋权的编号，不启动优化。"""
    return await invoke(prepare_request, request)


@app.post("/optimize")
async def optimize(request: DecisionRequest):
    """执行完整流程；NSGA-II与精确枚举使用同一业务评价器。"""
    return await invoke(optimize_request, request)


async def invoke_package(request: Request, function, total_budget=None):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_PACKAGE_BYTES:
            raise HTTPException(status_code=413, detail="资料包超过50MB。")
    try:
        payload = await run_in_threadpool(read_input_package, bytes(content))
        if total_budget is not None:
            payload.setdefault("configuration", {}).setdefault("solver", {})["total_budget"] = total_budget
        return await run_in_threadpool(function, payload)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("资料包校验失败：%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/prepare-package")
async def prepare_package(request: Request):
    """请求体为application/zip：校验分文件资料并生成单元。"""
    return await invoke_package(request, prepare_request)


@app.post("/optimize-package")
async def optimize_package(request: Request, view: str = "json", total_budget: float | None = Query(None, ge=0)):
    """请求体为application/zip：给资料，计算并返回完整空间方案。"""
    if view not in {"json", "html"}:
        raise HTTPException(status_code=422, detail="view仅支持json或html。")
    result = await invoke_package(request, optimize_request, total_budget)
    if view == "html":
        if result["status"] != "completed":
            raise HTTPException(status_code=422, detail=result["summary"]["message"])
        from model.geographic_map import result_map_html
        return HTMLResponse(await run_in_threadpool(result_map_html, result))
    return result


@app.get("/integration", response_class=HTMLResponse)
def integration_page():
    """宿主集成流程：输入资料、点击计算、显示结果，不依赖Streamlit。"""
    return (Path(__file__).parent/"assets/integration_client.html").read_text("utf-8")


@app.get("/input-template")
def input_template():
    return Response((Path(__file__).parent/"data/v1_1/peitian_input_package.zip").read_bytes(),
                    media_type="application/zip", headers={"Content-Disposition":'attachment; filename="peitian_input_package.zip"'})


@app.get("/symbols/{measure_id}.svg")
def measure_symbol(measure_id: str):
    from model.engineering_symbols import SYMBOL_NAMES, symbol_svg
    if measure_id not in SYMBOL_NAMES:
        raise HTTPException(status_code=404, detail="未知措施符号。")
    return Response(symbol_svg(measure_id), media_type="image/svg+xml")


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logger.error("未预期计算异常", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={"detail": "计算出现未预期异常，详情已记录在服务日志。"})
