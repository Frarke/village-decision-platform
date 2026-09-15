"""启动：python -m uvicorn api:app --host 127.0.0.1 --port 8000。"""
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from model.service import optimize_request, prepare_request


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


app = FastAPI(title="村庄地质灾害综合整治决策服务", version="1.0.0",
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


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logger.error("未预期计算异常", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=500, content={"detail": "计算出现未预期异常，详情已记录在服务日志。"})
