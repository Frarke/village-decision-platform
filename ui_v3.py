from __future__ import annotations

from io import BytesIO
from pathlib import Path
import copy
import hashlib
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.benchmarks import (
    BENCHMARK_SPECS,
    benchmark_functions_for_objectives,
    benchmark_spec_table,
    validate_standard_functions,
)
from model.data_validation import normalize_measures_frame, normalize_units_frame, validate_project_data
from model.evaluation import evaluate_unit
from model.i18n import FIELD_LABELS, chinese_frame
from model.recommendation import recommend
from model.schemas import GovernanceUnit, Measure, SolverConfig
from model.screening import screen_all
from model.solver import parameters_and_assumptions, result_to_json, solve_exact
from model.nsga2 import solve_nsga2
from model.spatial import STAGE_CONTROL_MEASURES, coordination_metrics, load_geojson, load_topology, polygon_centroid, spatial_unit_table
from model.tagging import tagged_unit_record
from model.validation import validate_runs
from model.workflow import GUIDED_STEPS, STEP_OUTPUTS, extract_unit_id, next_step, optimization_hash, preference_hash, step_validation, synchronize_selected_unit


DATA_DIR = ROOT / "data"
MODULES = ("阶段一 治理对象与基础数据", "阶段二 多维综合评估", "阶段三 多目标优化求解", "阶段四 方案评审与决策输出")
STEP_MODULE = (
    "阶段一 治理对象与基础数据",
    "阶段一 治理对象与基础数据",
    "阶段二 多维综合评估",
    "阶段二 多维综合评估",
    "阶段三 多目标优化求解",
    "阶段三 多目标优化求解",
    "阶段四 方案评审与决策输出",
    "阶段四 方案评审与决策输出",
)
STEP_STAGE_TITLE = {
    0: "阶段一：治理对象与基础数据",
    1: "阶段一：治理对象与基础数据",
    2: "阶段二：多维综合评估",
    3: "阶段二：多维综合评估",
    4: "阶段三：多目标优化求解",
    5: "阶段三：多目标优化求解",
    6: "阶段四：方案评审与决策输出",
    7: "阶段四：方案评审与决策输出",
}
COLORS = {"navy": "#102A43", "blue": "#173F5F", "cyan": "#168AAD", "muted": "#6B7C8F", "paper": "#F4F7FA", "red": "#C0392B", "orange": "#D97706", "green": "#2E7D5B"}
RISK_COLORS = {1: "#DDEFE5", 2: "#A9D8C0", 3: "#F2C66D", 4: "#E78856", 5: "#C0392B"}
STEP_DESCRIPTIONS = {
    0: "加载项目、研究区和基础资料，仅展示基础底图与数据完整性，不提前展开正式治理单元结论。",
    1: "导入、识别、补充和确认治理单元，形成可进入准入筛选的治理单元库。",
    2: "按照治理单元与措施库执行适用性准入筛选，查看准入依据、排除原因和统计结果。",
    3: "配置四目标参数、约束条件、风险权重与区域协同关系，为候选方案评价做准备。",
    4: "调用统一评价链执行方案集生成、精确枚举或NSGA-II多目标优化计算。",
    5: "查看Pareto候选方案、算法验证和代表方案选择依据。",
    6: "解析已经参与优化计算的村域协同结果、关键路径覆盖和代表方案空间表现。",
    7: "执行专家复核、推荐方案确认和成果导出。",
}

HOME_PAGE_ID = "home"
QUICK_PAGE_ID = "quick"
VALIDATION_PAGE_ID = "validation"
GUIDED_PAGE_IDS = tuple(f"guided_step_{index}" for index in range(len(GUIDED_STEPS)))
STAGE_IDS = ("stage1", "stage2", "stage3", "stage4")
STAGE_LABELS = dict(zip(STAGE_IDS, MODULES, strict=True))
STEP_TO_STAGE_ID = {
    0: "stage1",
    1: "stage1",
    2: "stage2",
    3: "stage2",
    4: "stage3",
    5: "stage3",
    6: "stage4",
    7: "stage4",
}
STAGE_TO_STEPS = {
    "stage1": (0, 1),
    "stage2": (2, 3),
    "stage3": (4, 5),
    "stage4": (6, 7),
}
QUICK_DATA_SOURCES = (
    "内置示例场景（开发示例）",
    "当前已确认治理单元数据",
    "上传数据文件",
)
SCENARIO_SETTING_KEYS = (
    "scenario_parameter_source",
    "budget_unit",
    "safety_unit",
    "total_budget",
    "min_fs",
    "min_safety",
    "max_measures",
    "allow_untreated",
    "max_enumerations",
    "drainage_synergy",
    "second_factor",
    "solver_method",
    "population_size",
    "generations",
    "crossover_probability",
    "mutation_probability",
    "random_seed",
    "weight_safety",
    "weight_economy",
    "weight_sustainability",
    "weight_resilience",
)
FORMAL_MEASURE_LIBRARY = (
    {"name": "截排水工程", "target": "坡面汇水、浅层滑坡、沟谷汇流", "function": "削减坡面入渗与集中汇流", "limit": "需核查排水出路与下游影响", "status": "科技报告正式措施库"},
    {"name": "灌草种植", "target": "浅层坡面、生态修复区", "function": "提升表层固土与水土保持能力", "limit": "对深层滑动面控制能力有限", "status": "科技报告正式措施库"},
    {"name": "植被混凝土", "target": "坡面防护与生态恢复", "function": "兼顾护坡与植被恢复", "limit": "需结合坡面稳定条件复核", "status": "科技报告正式措施库"},
    {"name": "格构锚固", "target": "中高陡坡面与破碎岩土体", "function": "提升坡面整体抗滑与约束能力", "limit": "需核查锚固深度与地层条件", "status": "科技报告正式措施库"},
    {"name": "挡墙工程", "target": "坡脚受限与建构筑物近邻区", "function": "提供坡脚支挡与局部稳定控制", "limit": "需校核基础承载与排水配套", "status": "科技报告正式措施库"},
    {"name": "抗滑桩", "target": "中深层滑坡与高风险控制区", "function": "增强主滑体抗滑承载能力", "limit": "需明确滑面、嵌固段与施工条件", "status": "科技报告正式措施库"},
    {"name": "谷坊坝", "target": "沟谷输移通道与泥石流物源沟", "function": "削减输移能力并滞蓄固体物质", "limit": "需与上游物源控制和下游行洪统筹", "status": "科技报告正式措施库"},
    {"name": "柔性拦挡结构", "target": "沟口、道路与承灾体防护区", "function": "拦截块石与浅表碎屑输移", "limit": "需核查拦挡等级与维护条件", "status": "科技报告正式措施库"},
)
HOME_STAGE_SUMMARY = (
    ("阶段一：治理单元确定", "项目与研究区设置、原始调查数据导入、治理单元识别、属性补充、地图复核和数据确认。"),
    ("阶段二：多维综合评估", "适用性初筛、四维评价、地形地貌与工程适宜性修正、区域协同关系配置与评估。"),
    ("阶段三：多目标优化求解", "方案组合规则、预算与安全底线约束、精确枚举或 NSGA-II 多目标优化、Pareto 候选分析。"),
    ("阶段四：方案评审与决策输出", "五类代表方案、协同结果解释、专家复核、推荐方案确认和成果导出。"),
)


def inject_style() -> None:
    st.markdown("""
    <style>
    :root{--navy:#102A43;--blue:#173F5F;--cyan:#168AAD;--green:#2E7D5B;--orange:#D97706;--red:#C0392B;--line:#D8E2EA}
    html,body,[class*="css"]{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif}
    .stApp{background:#EEF3F7}.block-container{max-width:1760px;padding:0.8rem 1.15rem 2.3rem}
    [data-testid="stSidebar"]{background:#F8FAFC;border-right:1px solid var(--line)}
    [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--navy)}
    h1{font-size:1.48rem!important;color:#F7FBFF!important;margin:0!important}h2{font-size:1.28rem!important;color:var(--navy)}h3{font-size:1.08rem!important;color:var(--blue)}
    .platform-head{background:var(--navy);border-radius:10px;padding:.75rem 1rem;color:#E7F0F7;box-shadow:0 3px 14px rgba(16,42,67,.14);margin-bottom:.55rem}
    .platform-sub{font-size:.82rem;color:#B9CBD9;margin-top:.2rem}.meta-line{font-size:.78rem;color:#D8E7EF;text-align:right;line-height:1.7}
    .work-card{background:#FFF;border:1px solid var(--line);border-radius:9px;padding:.78rem .9rem;box-shadow:0 2px 8px rgba(16,42,67,.05);margin:.35rem 0 .7rem}
    .task-card{background:#EAF3F7;border-left:4px solid var(--cyan);border-radius:7px;padding:.68rem .8rem;margin:.35rem 0 .7rem;color:#28485F}
    .notice{background:#FFF8E8;border-left:4px solid var(--orange);border-radius:7px;padding:.68rem .8rem;color:#60420B;margin:.35rem 0 .7rem}
    .invalid{background:#FDECEC;border-left:4px solid var(--red);border-radius:7px;padding:.72rem .85rem;color:#7B241C;font-weight:600;margin:.35rem 0 .7rem}
    .ok{background:#EAF6F0;border-left:4px solid var(--green);border-radius:7px;padding:.68rem .8rem;color:#1E5A41;margin:.35rem 0 .7rem}
    .step-list{font-size:.84rem;line-height:1.9;color:#536B7D}.step-now{font-weight:700;color:var(--cyan)}.step-done{color:var(--green)}
    .section-label{font-size:.72rem;letter-spacing:.08em;color:#6B7C8F;font-weight:700;margin:.25rem 0 .1rem}
    [data-testid="stMetric"]{background:#FFF;border:1px solid var(--line);border-radius:8px;padding:.62rem .72rem;box-shadow:none}
    [data-testid="stMetricValue"]{font-size:1.12rem}.stButton>button{border-radius:6px}.stButton>button[kind="primary"]{background:var(--cyan);border-color:var(--cyan)}
    [data-testid="stDataFrame"]{background:#FFF;border:1px solid var(--line);border-radius:8px;overflow:hidden}
    div[data-baseweb="tab-list"]{gap:.2rem}div[data-baseweb="tab"]{height:2.35rem}
    .legend-row{display:flex;gap:.75rem;flex-wrap:wrap;font-size:.76rem;color:#566B7A}.legend-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
    @media(max-width:1200px){.block-container{padding-left:.7rem;padding-right:.7rem}.meta-line{text-align:left}}
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_defaults() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    return (
        pd.read_csv(DATA_DIR / "default_units.csv"),
        pd.read_csv(DATA_DIR / "default_measures.csv"),
        load_geojson(DATA_DIR / "spatial_units.geojson"),
        load_geojson(DATA_DIR / "spatial_context.geojson"),
        load_topology(DATA_DIR / "spatial_topology.json"),
        json.loads((DATA_DIR / "default_scenario.json").read_text(encoding="utf-8")),
    )


def state_defaults() -> dict[str, object]:
    units, measures, units_geojson, context_geojson, topology, scenario = load_defaults()
    return {
        "mode": "平台首页", "current_page_id": HOME_PAGE_ID, "pending_page_id": None, "current_step": 0, "current_module": MODULES[0], "completed_steps": set(),
        "intelligence_page": "工程措施准入", "last_intelligence_step": -1,
        "project_name": scenario["project_name"], "village_name": scenario["village_name"], "data_version": scenario["data_version"],
        "scenario_parameter_source": scenario["parameter_source"], "budget_unit": scenario["budget_unit"], "safety_unit": scenario["safety_unit"],
        "data_source_label": "内置示例场景（开发示例）", "data_source_kind": "example", "quick_data_source": QUICK_DATA_SOURCES[0],
        "units_df": units.copy(), "measures_df": measures.copy(), "units_geojson": copy.deepcopy(units_geojson),
        "context_geojson": copy.deepcopy(context_geojson), "topology": copy.deepcopy(topology), "editor_version": 0,
        "selected_unit": "U01", "unit_selector_widget": "U01", "pending_unit_selector": None, "selected_plan_label": "综合折中代表方案",
        "map_background": "卫星影像", "display_mode": "治理前风险", "show_paths": False, "show_units_on_overview": False,
        "result": None, "exact_result": None, "nsga_result": None, "validation": None, "benchmark_results": None,
        "result_valid": False, "result_invalid_reason": "", "last_optimization_hash": None, "last_preference_hash": None, "scenario_history": [],
        "total_budget": scenario["total_budget"], "max_measures": scenario["max_measures"], "allow_untreated": scenario["allow_untreated"], "min_fs": scenario["min_fs"], "min_safety": scenario["min_safety"],
        "max_enumerations": scenario["max_enumerations"], "drainage_synergy": scenario["drainage_synergy"], "second_factor": scenario["second_factor"],
        "weight_safety": scenario["weight_safety"], "weight_economy": scenario["weight_economy"], "weight_sustainability": scenario["weight_sustainability"], "weight_resilience": scenario["weight_resilience"],
        "solver_method": scenario["solver_method"], "population_size": scenario["population_size"], "generations": scenario["generations"], "crossover_probability": scenario["crossover_probability"],
        "mutation_probability": scenario["mutation_probability"], "random_seed": scenario["random_seed"], "verify_current_run": False,
        "benchmark_mode": "single", "benchmark_objectives": 2, "benchmark_function": "ZDT1", "benchmark_variables": 30,
        "benchmark_population": 80, "benchmark_generations": 80, "benchmark_seed": 42, "benchmark_batch_seeds": "42, 123, 2026",
        "benchmark_batch_functions": ["ZDT1"],
        "guided_snapshot": None,
        "pending_scenario": None,
        "form_revision": 0,
    }


def initialize_state() -> None:
    for key, value in state_defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


def page_id_for_step(step: int) -> str:
    return GUIDED_PAGE_IDS[int(step)]


def step_for_page_id(page_id: str) -> int | None:
    if page_id in GUIDED_PAGE_IDS:
        return GUIDED_PAGE_IDS.index(page_id)
    return None


def current_stage_id() -> str:
    return STEP_TO_STAGE_ID[int(st.session_state.current_step)]


def is_home_mode() -> bool:
    return st.session_state.mode == "平台首页"


def is_guided_mode() -> bool:
    return st.session_state.mode == "引导式研判"


def is_quick_mode() -> bool:
    return st.session_state.mode == "快速分析"


def is_validation_mode() -> bool:
    return st.session_state.mode == "算法验证中心"


def apply_page_state(page_id: str) -> None:
    st.session_state.current_page_id = page_id
    if page_id == HOME_PAGE_ID:
        st.session_state.mode = "平台首页"
        return
    if page_id == QUICK_PAGE_ID:
        st.session_state.mode = "快速分析"
        return
    if page_id == VALIDATION_PAGE_ID:
        st.session_state.mode = "算法验证中心"
        return
    step = step_for_page_id(page_id)
    if step is None:
        return
    st.session_state.mode = "引导式研判"
    st.session_state.current_step = step
    st.session_state.current_module = STEP_MODULE[step]


def go_to_page(page_id: str) -> None:
    st.session_state.pending_page_id = page_id
    st.rerun()


def go_to_guided_step(step: int) -> None:
    go_to_page(page_id_for_step(step))


def switch_mode(mode: str) -> None:
    if mode == "平台首页":
        go_to_page(HOME_PAGE_ID)
    elif mode == "快速分析":
        go_to_page(QUICK_PAGE_ID)
    elif mode == "算法验证中心":
        go_to_page(VALIDATION_PAGE_ID)
    else:
        go_to_guided_step(int(st.session_state.current_step))


def apply_pending_navigation() -> None:
    pending = st.session_state.pending_page_id
    if pending:
        apply_page_state(str(pending))
        st.session_state.pending_page_id = None
    else:
        apply_page_state(str(st.session_state.current_page_id))


def update_data_source(source_label: str, source_kind: str) -> None:
    st.session_state.data_source_label = source_label
    st.session_state.data_source_kind = source_kind


def queue_scenario(
    units_df: pd.DataFrame,
    measures_df: pd.DataFrame,
    geojson: dict,
    topology: dict,
    *,
    source_label: str,
    source_kind: str,
    project_name: str | None = None,
    village_name: str | None = None,
    data_version: str | None = None,
) -> None:
    st.session_state.pending_scenario = {
        "units_df": normalize_units_frame(units_df),
        "measures_df": normalize_measures_frame(measures_df),
        "units_geojson": copy.deepcopy(geojson),
        "topology": copy.deepcopy(topology),
        "source_label": source_label,
        "source_kind": source_kind,
        "project_name": project_name,
        "village_name": village_name,
        "data_version": data_version,
    }


def request_scenario_apply(payload: dict[str, object]) -> None:
    st.session_state.pending_scenario = payload
    st.rerun()


def data_snapshot() -> dict[str, object]:
    return {
        "units_df": st.session_state.units_df.copy(),
        "measures_df": st.session_state.measures_df.copy(),
        "units_geojson": copy.deepcopy(st.session_state.units_geojson),
        "topology": copy.deepcopy(st.session_state.topology),
        "project_name": str(st.session_state.project_name),
        "village_name": str(st.session_state.village_name),
        "data_version": str(st.session_state.data_version),
        "scenario_settings": {key: copy.deepcopy(st.session_state[key]) for key in SCENARIO_SETTING_KEYS},
    }


def default_scenario_payload() -> dict[str, object]:
    units, measures, units_geojson, _, topology, scenario = load_defaults()
    return {
        "units_df": normalize_units_frame(units),
        "measures_df": normalize_measures_frame(measures),
        "units_geojson": copy.deepcopy(units_geojson),
        "topology": copy.deepcopy(topology),
        "source_label": "内置示例场景（开发示例）",
        "source_kind": "example",
        "project_name": scenario["project_name"],
        "village_name": scenario["village_name"],
        "data_version": scenario["data_version"],
        "scenario_settings": {
            "scenario_parameter_source": scenario["parameter_source"],
            "budget_unit": scenario["budget_unit"],
            "safety_unit": scenario["safety_unit"],
            "total_budget": scenario["total_budget"],
            "min_fs": scenario["min_fs"],
            "min_safety": scenario["min_safety"],
            "max_measures": scenario["max_measures"],
            "allow_untreated": scenario["allow_untreated"],
            "max_enumerations": scenario["max_enumerations"],
            "drainage_synergy": scenario["drainage_synergy"],
            "second_factor": scenario["second_factor"],
            "solver_method": scenario["solver_method"],
            "population_size": scenario["population_size"],
            "generations": scenario["generations"],
            "crossover_probability": scenario["crossover_probability"],
            "mutation_probability": scenario["mutation_probability"],
            "random_seed": scenario["random_seed"],
            "weight_safety": scenario["weight_safety"],
            "weight_economy": scenario["weight_economy"],
            "weight_sustainability": scenario["weight_sustainability"],
            "weight_resilience": scenario["weight_resilience"],
        },
    }


def save_guided_snapshot() -> None:
    st.session_state.guided_snapshot = data_snapshot()


def load_snapshot(snapshot: dict[str, object], *, source_label: str, source_kind: str) -> None:
    apply_loaded_data(
        snapshot["units_df"],
        snapshot["measures_df"],
        snapshot["units_geojson"],
        snapshot["topology"],
        source_label=source_label,
        source_kind=source_kind,
        project_name=str(snapshot.get("project_name", st.session_state.project_name)),
        village_name=str(snapshot.get("village_name", st.session_state.village_name)),
        data_version=str(snapshot.get("data_version", st.session_state.data_version)),
        scenario_settings=copy.deepcopy(snapshot.get("scenario_settings")),
    )


def restore_example_scenario() -> None:
    request_scenario_apply(default_scenario_payload())


def apply_pending_scenario() -> None:
    pending = st.session_state.pending_scenario
    if not pending:
        return
    st.session_state.pending_scenario = None
    apply_loaded_data(
        pending["units_df"],
        pending["measures_df"],
        pending["units_geojson"],
        pending["topology"],
        source_label=str(pending["source_label"]),
        source_kind=str(pending["source_kind"]),
        project_name=str(pending["project_name"]) if pending.get("project_name") is not None else None,
        village_name=str(pending["village_name"]) if pending.get("village_name") is not None else None,
        data_version=str(pending["data_version"]) if pending.get("data_version") is not None else None,
        scenario_settings=copy.deepcopy(pending.get("scenario_settings")),
    )
    st.session_state.form_revision += 1
    st.session_state.editor_version += 1


def reset_defaults() -> None:
    for key, value in state_defaults().items():
        st.session_state[key] = value
    st.session_state.editor_version += 1


def make_config() -> SolverConfig:
    return SolverConfig(
        total_budget=float(st.session_state.total_budget), max_measures_per_unit=int(st.session_state.max_measures),
        allow_untreated_medium_high=bool(st.session_state.allow_untreated), high_risk_min_fs=float(st.session_state.min_fs),
        high_risk_min_safety=float(st.session_state.min_safety), max_enumerations=int(st.session_state.max_enumerations),
        drainage_synergy=float(st.session_state.drainage_synergy), general_second_measure_factor=float(st.session_state.second_factor),
        nsga_population_size=int(st.session_state.population_size), nsga_generations=int(st.session_state.generations),
        nsga_crossover_probability=float(st.session_state.crossover_probability),
        nsga_mutation_probability=float(st.session_state.mutation_probability), random_seed=int(st.session_state.random_seed),
        preference_weights={"safety": st.session_state.weight_safety, "economy": st.session_state.weight_economy,
                            "sustainability": st.session_state.weight_sustainability, "resilience": st.session_state.weight_resilience},
    )


def model_inputs() -> tuple[list[GovernanceUnit], list[Measure]]:
    units = [GovernanceUnit.from_record(row) for row in st.session_state.units_df.to_dict("records")]
    measures = [Measure.from_record(row) for row in st.session_state.measures_df.to_dict("records")]
    return units, measures


def current_optimization_hash() -> str:
    units, measures = model_inputs()
    return optimization_hash(units, measures, make_config(), st.session_state.solver_method, st.session_state.topology)


def current_preference_hash() -> str:
    return preference_hash(make_config().preference_weights)


def rebuild_representatives(result) -> None:
    if result is None or result.pareto.empty:
        return
    result.representatives = recommend(result.pareto, make_config().preference_weights)
    rows: list[dict[str, object]] = []
    for _, recommendation in result.representatives.iterrows():
        plan_id = str(recommendation["plan_id"])
        for detail in result.detail_lookup[plan_id]:
            rows.append({"recommendation_type": recommendation["recommendation_type"], "plan_id": plan_id, **detail})
    result.representative_details = pd.DataFrame(rows)


def refresh_result_state() -> None:
    if st.session_state.result is None:
        return
    try:
        if st.session_state.last_optimization_hash != current_optimization_hash():
            invalidate_results("输入数据、约束条件或求解方式已变化，需重新计算。")
            return
        new_pref = current_preference_hash()
        if st.session_state.last_preference_hash != new_pref:
            for result in (st.session_state.exact_result, st.session_state.nsga_result):
                rebuild_representatives(result)
            st.session_state.result = st.session_state.exact_result if st.session_state.solver_method == "exact" else st.session_state.nsga_result
            st.session_state.last_preference_hash = new_pref
            st.session_state.result_invalid_reason = ""
    except (KeyError, TypeError, ValueError):
        invalidate_results("当前结果与输入状态不一致，需重新计算。")


def editor_columns(frame: pd.DataFrame) -> dict:
    configs = {}
    for name in frame.columns:
        label = FIELD_LABELS.get(name, name)
        if pd.api.types.is_bool_dtype(frame[name]):
            configs[name] = st.column_config.CheckboxColumn(label)
        elif pd.api.types.is_numeric_dtype(frame[name]):
            configs[name] = st.column_config.NumberColumn(label)
        elif name == "evidence_level":
            configs[name] = st.column_config.SelectboxColumn(label, options=["科技报告依据", "规范依据", "文献或案例依据", "专家经验参数", "待标定参数", "示例参数"])
        else:
            configs[name] = st.column_config.TextColumn(label)
    return configs


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def render_frame(frame: pd.DataFrame, *, use_container_width: bool = True, hide_index: bool = True) -> None:
    try:
        st.dataframe(frame, use_container_width=use_container_width, hide_index=hide_index)
    except Exception:
        fallback = frame if not hide_index else frame.reset_index(drop=True)
        st.markdown(fallback.to_html(index=not hide_index, escape=False), unsafe_allow_html=True)
        st.caption("当前运行环境未能启用交互式表格，已自动切换为静态表格回退。")


def render_step_banner() -> None:
    if not is_guided_mode():
        return
    step = int(st.session_state.current_step)
    st.markdown(
        f"<div class='work-card'><div class='section-label'>{STEP_STAGE_TITLE[step]}</div>"
        f"<h2 style='margin:.1rem 0 .35rem 0'>{step + 1}. {GUIDED_STEPS[step]}</h2>"
        f"<div style='color:#546A7B;font-size:.92rem'>{STEP_DESCRIPTIONS[step]}</div></div>",
        unsafe_allow_html=True,
    )


def invalidate_results(reason: str | None = None) -> None:
    st.session_state.result_valid = False
    if reason:
        st.session_state.result_invalid_reason = reason


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError(f"暂不支持的表格格式：{suffix}")


def read_uploaded_json(uploaded_file) -> dict:
    if uploaded_file is None:
        raise ValueError("未提供 JSON 文件。")
    return json.load(uploaded_file)


def field_completeness(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(frame.notna().mean().mean() * 100.0)


def spatial_bounds_summary() -> str:
    table = spatial_unit_table(st.session_state.units_geojson)
    if table.empty:
        return "无有效坐标"
    return (
        f"经度 {table.longitude.min():.5f}～{table.longitude.max():.5f}；"
        f"纬度 {table.latitude.min():.5f}～{table.latitude.max():.5f}"
    )


def current_data_summary() -> dict[str, object]:
    return {
        "project_name": str(st.session_state.project_name).strip(),
        "village_name": str(st.session_state.village_name).strip(),
        "data_version": str(st.session_state.data_version).strip(),
        "data_source": st.session_state.data_source_label,
        "unit_count": int(len(st.session_state.units_df)),
        "path_count": int(len(st.session_state.topology.get("paths", []))),
        "field_completeness": field_completeness(st.session_state.units_df),
        "spatial_bounds": spatial_bounds_summary(),
        "measure_count": int(len(st.session_state.measures_df)),
        "parameter_status": st.session_state.scenario_parameter_source,
    }


def selected_quick_source_kind() -> str:
    source = st.session_state.quick_data_source
    if source == QUICK_DATA_SOURCES[0]:
        return "example"
    if source == QUICK_DATA_SOURCES[1]:
        return "guided"
    return "upload"


def quick_analysis_requirements(required_source_kind: str | None = None) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    summary = current_data_summary()
    expected_kind = required_source_kind or selected_quick_source_kind()
    if st.session_state.data_source_kind != expected_kind:
        labels = {
            "example": "内置示例场景",
            "guided": "引导式研判已确认数据",
            "upload": "上传数据文件",
        }
        blockers.append(f"当前尚未实际载入{labels[expected_kind]}")
    if not summary["project_name"]:
        blockers.append("缺少项目名称")
    if not summary["village_name"]:
        blockers.append("缺少研究区名称")
    if not summary["data_version"]:
        blockers.append("缺少数据版本")
    if summary["unit_count"] <= 0:
        blockers.append("缺少治理单元数据")
    if summary["measure_count"] <= 0:
        blockers.append("缺少工程措施库")
    if summary["path_count"] <= 0:
        blockers.append("缺少风险路径数据")
    blockers.extend(validate_project_data(st.session_state.units_df, st.session_state.measures_df, st.session_state.units_geojson, st.session_state.topology))
    return not blockers, blockers


def apply_loaded_data(
    units_df: pd.DataFrame,
    measures_df: pd.DataFrame,
    geojson: dict,
    topology: dict,
    *,
    source_label: str,
    source_kind: str,
    project_name: str | None = None,
    village_name: str | None = None,
    data_version: str | None = None,
    scenario_settings: dict[str, object] | None = None,
) -> None:
    units_df = normalize_units_frame(units_df)
    measures_df = normalize_measures_frame(measures_df)
    st.session_state.units_df = units_df.copy()
    st.session_state.measures_df = measures_df.copy()
    st.session_state.units_geojson = copy.deepcopy(geojson)
    st.session_state.topology = copy.deepcopy(topology)
    st.session_state.selected_unit = synchronize_selected_unit(st.session_state.selected_unit, units_df.unit_id.astype(str).tolist())
    if project_name is not None:
        st.session_state.project_name = project_name
    if village_name is not None:
        st.session_state.village_name = village_name
    if data_version is not None:
        st.session_state.data_version = data_version
    if scenario_settings:
        for key in SCENARIO_SETTING_KEYS:
            if key in scenario_settings:
                st.session_state[key] = copy.deepcopy(scenario_settings[key])
    update_data_source(source_label, source_kind)
    if source_kind == "guided":
        save_guided_snapshot()
    invalidate_results("数据或空间拓扑已更新，需要重新计算。")


def quick_form_widget_keys() -> dict[str, str]:
    revision = int(st.session_state.form_revision)
    return {
        "project_name": f"quick_project_name_{revision}",
        "village_name": f"quick_village_name_{revision}",
        "data_version": f"quick_data_version_{revision}",
        "solver_method": f"quick_solver_method_{revision}",
        "total_budget": f"quick_total_budget_{revision}",
        "min_fs": f"quick_min_fs_{revision}",
        "min_safety": f"quick_min_safety_{revision}",
        "max_measures": f"quick_max_measures_{revision}",
        "allow_untreated": f"quick_allow_untreated_{revision}",
    }


def initialize_quick_form_state() -> dict[str, str]:
    mapping = quick_form_widget_keys()
    for state_key, widget_key in mapping.items():
        if widget_key not in st.session_state:
            st.session_state[widget_key] = copy.deepcopy(st.session_state[state_key])
    return mapping


def sync_quick_form_state(mapping: dict[str, str]) -> list[str]:
    changed: list[str] = []
    for state_key, widget_key in mapping.items():
        widget_value = st.session_state[widget_key]
        if st.session_state[state_key] != widget_value:
            st.session_state[state_key] = copy.deepcopy(widget_value)
            changed.append(state_key)
    return changed


def selected_plan(result) -> tuple[str, list[dict[str, object]]]:
    if result is None or result.representatives.empty:
        return "", []
    labels = result.representatives["recommendation_type"].astype(str).tolist()
    if st.session_state.selected_plan_label not in labels:
        st.session_state.selected_plan_label = "综合折中代表方案" if "综合折中代表方案" in labels else labels[0]
    row = result.representatives[result.representatives["recommendation_type"] == st.session_state.selected_plan_label].iloc[0]
    plan_id = str(row["plan_id"])
    return plan_id, result.detail_lookup.get(plan_id, [])


def status_chip(text: str, tone: str = "ok") -> str:
    palette = {
        "ok": ("#EAF6F0", "#1E5A41", "#2E7D5B"),
        "warn": ("#FFF8E8", "#7A5310", "#D97706"),
        "bad": ("#FDECEC", "#7B241C", "#C0392B"),
        "info": ("#EAF3F7", "#28485F", "#168AAD"),
    }
    bg, fg, bd = palette[tone]
    return f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;background:{bg};color:{fg};border:1px solid {bd};font-size:.78rem;margin-right:6px'>{text}</span>"


def map_frame() -> pd.DataFrame:
    spatial = spatial_unit_table(st.session_state.units_geojson)
    return st.session_state.units_df.merge(spatial, on="unit_id", how="left")


def path_stage_status(details: dict[str, dict[str, object]], path: dict[str, object]) -> list[bool]:
    ids = [str(path["source_unit"]), str(path["transport_unit"]), str(path["receptor_unit"])]
    status = []
    for stage, unit_id in zip(("source", "transport", "receptor"), ids, strict=True):
        measures = set(str(details.get(unit_id, {}).get("measures", "不治理")).split("＋"))
        status.append(bool(measures & STAGE_CONTROL_MEASURES[stage]))
    return status


def _collect_geometry_points(geometry: dict) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    geo_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geo_type == "Point":
        points.append((float(coords[0]), float(coords[1])))
    elif geo_type == "LineString":
        points.extend((float(item[0]), float(item[1])) for item in coords)
    elif geo_type == "Polygon":
        for ring in coords:
            points.extend((float(item[0]), float(item[1])) for item in ring)
    elif geo_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                points.extend((float(item[0]), float(item[1])) for item in ring)
    return points


def map_extent(show_units: bool = True) -> tuple[dict[str, float], float]:
    points: list[tuple[float, float]] = []
    if show_units:
        table = spatial_unit_table(st.session_state.units_geojson)
        points.extend((float(row.longitude), float(row.latitude)) for row in table.itertuples())
    for feature in st.session_state.context_geojson.get("features", []):
        points.extend(_collect_geometry_points(feature.get("geometry", {})))
    if not points:
        table = spatial_unit_table(st.session_state.units_geojson)
        points.extend((float(row.longitude), float(row.latitude)) for row in table.itertuples())
    lons = [item[0] for item in points]
    lats = [item[1] for item in points]
    lon_span = max(max(lons) - min(lons), 1e-4)
    lat_span = max(max(lats) - min(lats), 1e-4)
    span = max(lon_span, lat_span)
    if span > 1.0:
        zoom = 8.0
    elif span > 0.5:
        zoom = 9.0
    elif span > 0.2:
        zoom = 10.0
    elif span > 0.1:
        zoom = 11.2
    elif span > 0.05:
        zoom = 12.0
    elif span > 0.02:
        zoom = 13.0
    elif span > 0.01:
        zoom = 14.0
    else:
        zoom = 15.0
    return {"lon": float((min(lons) + max(lons)) / 2.0), "lat": float((min(lats) + max(lats)) / 2.0)}, zoom


def _map_style_and_layers(background: str) -> tuple[bool, str, list[dict[str, object]], str]:
    if background == "卫星影像":
        return True, "white-bg", [{"below": "traces", "sourcetype": "raster", "sourceattribution": "Esri World Imagery", "source": ["https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}], "Esri World Imagery"
    if background == "卫星影像+标注":
        return True, "satellite-streets", [], "Mapbox Satellite Streets"
    if background == "街道地图":
        return True, "open-street-map", [], "OpenStreetMap"
    if background == "地形图":
        return True, "carto-positron", [], "Carto"
    return False, "white-bg", [], "离线坐标网格"


def map_figure(plan_details: list[dict[str, object]] | None, selected_unit: str, online: bool, show_units: bool = True) -> go.Figure:
    frame = map_frame()
    detail_map = {str(item["unit_id"]): item for item in (plan_details or [])}
    mode = st.session_state.display_mode
    if mode == "治理前风险":
        frame["display"] = frame["risk"]; label = "治理前风险等级"; scale = [[0, "#DDEFE5"], [.5, "#F2C66D"], [1, "#C0392B"]]
    elif mode == "治理后安全性":
        frame["display"] = frame["unit_id"].map(lambda x: float(detail_map.get(str(x), {}).get("safety", 0)))
        label = "治理后安全得分"; scale = [[0, "#C0392B"], [.55, "#F2C66D"], [1, "#2E7D5B"]]
    elif mode == "推荐措施":
        frame["display"] = frame["unit_id"].map(lambda x: 0 if detail_map.get(str(x), {}).get("measures", "不治理") == "不治理" else 1)
        label = "是否配置措施"; scale = [[0, "#D7DEE4"], [.49, "#D7DEE4"], [.5, "#168AAD"], [1, "#173F5F"]]
    else:
        frame["display"] = frame["unit_id"].map(lambda x: 1 if detail_map and float(detail_map.get(str(x), {}).get("safety", 0)) < st.session_state.min_safety else 0)
        label = "约束状态"; scale = [[0, "#2E7D5B"], [.49, "#2E7D5B"], [.5, "#C0392B"], [1, "#C0392B"]]
    center, zoom = map_extent(show_units=show_units)
    fig = go.Figure()
    scatter = go.Scattermap if online else go.Scattergeo
    if show_units:
        common = dict(geojson=st.session_state.units_geojson, locations=frame.unit_id, z=frame.display,
                      featureidkey="properties.unit_id", colorscale=scale, marker_line_width=1.6,
                      customdata=np.stack([frame.unit_id, frame.unit_name], axis=-1),
                      hovertemplate="治理单元：%{customdata[0]}<br>%{customdata[1]}<br>" + label + "：%{z:.1f}<extra></extra>")
        if online:
            fig.add_trace(go.Choroplethmap(**common, marker_opacity=.55, colorbar_title=label))
        else:
            fig.add_trace(go.Choropleth(**common, colorbar_title=label))
        sizes = [18 if str(uid) == selected_unit else 12 for uid in frame.unit_id]
        colors = [COLORS["orange"] if str(uid) == selected_unit else "#FFFFFF" for uid in frame.unit_id]
        selection_marker = {"size": sizes, "color": colors}
        if not online:
            selection_marker["line"] = {"color": COLORS["navy"], "width": 2}
        fig.add_trace(scatter(lon=frame.longitude, lat=frame.latitude, mode="markers+text", text=frame.unit_id,
                              textposition="top center", marker=selection_marker,
                              customdata=np.stack([frame.unit_id, frame.unit_name], axis=-1),
                              hovertemplate="点击选择：%{customdata[0]}<br>%{customdata[1]}<extra></extra>", name="治理单元选择点"))
        selected_feature = next((f for f in st.session_state.units_geojson["features"] if str(f["properties"].get("unit_id")) == selected_unit), None)
        if selected_feature:
            ring = selected_feature["geometry"]["coordinates"][0]
            fig.add_trace(scatter(lon=[p[0] for p in ring], lat=[p[1] for p in ring], mode="lines",
                                  line={"color": COLORS["orange"], "width": 5}, hoverinfo="skip", name="当前选中单元"))
    for feature in st.session_state.context_geojson.get("features", []):
        geometry = feature["geometry"]; category = feature["properties"].get("category", "空间要素"); name = feature["properties"].get("name", "未命名")
        if geometry["type"] == "LineString":
            coords = geometry["coordinates"]; color = "#2C7FB8" if "水系" in category else "#776B5D"
            fig.add_trace(scatter(lon=[p[0] for p in coords], lat=[p[1] for p in coords], mode="lines", line={"color": color, "width": 2}, name=f"{category}｜{name}", hoverinfo="name"))
        elif geometry["type"] == "Point":
            fig.add_trace(scatter(lon=[geometry["coordinates"][0]], lat=[geometry["coordinates"][1]], mode="markers", marker={"size": 10, "color": "#6D4C41"}, name=f"{category}｜{name}", hoverinfo="name"))
    if st.session_state.show_paths:
        for path in st.session_state.topology.get("paths", []):
            ids = [str(path["source_unit"]), str(path["transport_unit"]), str(path["receptor_unit"])]
            points = frame.set_index("unit_id").loc[ids]
            stages = path_stage_status(detail_map, path)
            color = COLORS["green"] if all(stages) else (COLORS["orange"] if any(stages) else COLORS["red"])
            fig.add_trace(scatter(lon=points.longitude, lat=points.latitude, mode="lines+markers+text",
                                  text=["物源", "输移", "承灾体"], textposition="bottom center",
                                  line={"color": color, "width": 3}, marker={"size": [9, 8, 11], "symbol": "circle"},
                                  name=f"{path['path_id']}｜{path['name']}｜物源→输移→承灾体", hoverinfo="name"))
    if online:
        _, style_name, layers, _ = _map_style_and_layers(st.session_state.map_background)
        fig.update_layout(map={"style": style_name, "center": center, "zoom": zoom, "layers": layers})
    else:
        fig.update_geos(fitbounds="locations", visible=True, showland=True, landcolor="#E8EEF2", showcoastlines=False,
                        lonaxis={"showgrid": True, "gridcolor": "#C9D5DE"}, lataxis={"showgrid": True, "gridcolor": "#C9D5DE"})
    title_text = "研究区基础底图与资料状态" if not show_units else f"村庄级空间研判｜{mode}"
    fig.update_layout(height=585, margin={"l": 0, "r": 0, "t": 32, "b": 0}, showlegend=False,
                      title={"text": title_text, "x": .01, "font": {"size": 15, "color": COLORS["navy"]}},
                      clickmode="event+select", uirevision="v3-map-view", selectionrevision="v3-map-selection")
    return fig


def render_map(plan_details: list[dict[str, object]] | None, key: str, show_units: bool = True) -> None:
    ids = st.session_state.units_df.unit_id.astype(str).tolist()
    st.session_state.selected_unit = synchronize_selected_unit(st.session_state.selected_unit, ids)
    online, _, _, _ = _map_style_and_layers(st.session_state.map_background)
    try:
        event = st.plotly_chart(map_figure(plan_details, st.session_state.selected_unit, online, show_units=show_units), use_container_width=True,
                                key=key, on_select="rerun", selection_mode="points", config={"displaylogo": False, "scrollZoom": True})
        clicked = extract_unit_id(event, ids) if show_units else None
        if show_units and clicked and clicked != st.session_state.selected_unit:
            st.session_state.selected_unit = clicked
            st.session_state.pending_unit_selector = clicked
            st.rerun()
    except Exception as exc:
        if online:
            st.warning(f"在线底图不可用，已自动切换到离线坐标网格。治理单元和属性仍可使用。原因：{exc}")
            try:
                event = st.plotly_chart(map_figure(plan_details, st.session_state.selected_unit, False, show_units=show_units), use_container_width=True,
                                        key=f"{key}_offline", on_select="rerun", selection_mode="points", config={"displaylogo": False, "scrollZoom": True})
                clicked = extract_unit_id(event, ids) if show_units else None
                if show_units and clicked and clicked != st.session_state.selected_unit:
                    st.session_state.selected_unit = clicked; st.session_state.pending_unit_selector = clicked; st.rerun()
            except Exception as offline_exc:
                st.error(f"本地空间图层无法绘制：{offline_exc}")
        else:
            st.error(f"本地空间图层无法绘制：{exc}")


def unit_detail(result, plan_details: list[dict[str, object]] | None) -> None:
    ids = st.session_state.units_df.unit_id.astype(str).tolist()
    selected = synchronize_selected_unit(st.session_state.selected_unit, ids)
    st.session_state.selected_unit = selected
    if st.session_state.pending_unit_selector:
        st.session_state.unit_selector_widget = st.session_state.pending_unit_selector
        st.session_state.pending_unit_selector = None
    elif st.session_state.unit_selector_widget != selected:
        st.session_state.unit_selector_widget = selected
    choice = st.selectbox("治理单元定位", ids, key="unit_selector_widget")
    if choice != st.session_state.selected_unit:
        st.session_state.selected_unit = choice
        st.rerun()
    row = st.session_state.units_df[st.session_state.units_df.unit_id.astype(str) == selected].iloc[0]
    spatial = spatial_unit_table(st.session_state.units_geojson).set_index("unit_id")
    props = spatial.loc[selected] if selected in spatial.index else pd.Series(dtype=object)
    detail = next((item for item in (plan_details or []) if str(item["unit_id"]) == selected), {})
    tagged = tagged_unit_record(GovernanceUnit.from_record(row.to_dict()))
    st.markdown(f"### {selected}｜{row['unit_name']}")
    st.caption("当前详情用于治理单元复核；原始调查点、识别建议和已确认治理单元应在流程中分层理解。")
    screening = None
    if result is not None:
        screening = result.screening[result.screening.unit_id.astype(str) == selected]
    else:
        units, measures = model_inputs(); frame = pd.DataFrame([item.to_dict() for item in screen_all(units, measures)])
        screening = frame[frame.unit_id.astype(str) == selected]
    admitted = "、".join(screening.loc[screening.eligible, "measure_name"].astype(str)) or "无"
    excluded = "；\n\n".join(f"{item.measure_name}：{item.exclusion_reason}" for item in screening.loc[~screening.eligible].itertuples()) or "无"
    tabs = st.tabs(["基本信息", "风险属性", "识别标签", "准入结果", "路径与依据"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.markdown(f"{status_chip('已确认治理单元', 'ok')}{status_chip(str(row['evidence_level']), 'info')}", unsafe_allow_html=True)
        c2.markdown(status_chip(f"编号 {selected}", "info"), unsafe_allow_html=True)
        grid1, grid2 = st.columns(2)
        grid1.metric("灾害类型", props.get("hazard_type", "待补充"))
        grid2.metric("主要承灾体", props.get("protected_object", "待补充"))
        grid1.metric("岩土体类型", props.get("material_type", "待补充"))
        grid2.metric("空间位置", f"{props.get('longitude', 0):.5f}, {props.get('latitude', 0):.5f}")
        st.markdown(f"<div class='work-card'><b>备注</b><br>{row['notes'] or '无'}</div>", unsafe_allow_html=True)
    with tabs[1]:
        a1, a2 = st.columns(2)
        a1.metric("风险等级", f"{int(row['risk'])}级")
        a2.metric("危险性等级", f"{int(row['haz'])}级")
        a1.metric("坡度", f"{row['slope_deg']:g}°")
        a2.metric("坡高", f"{row['height_m']:g} 米")
        a1.metric("与承灾体距离", f"{row['distance_m']:g} 米")
        a2.metric("现状稳定系数", f"{row['fs0']:.2f}")
    with tabs[2]:
        st.markdown(f"<div class='work-card'><b>自动识别标签</b><br>{tagged['auto_tags']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='work-card'><b>标签判定依据</b><br>{tagged['tag_evidence']}</div>", unsafe_allow_html=True)
    with tabs[3]:
        st.markdown(f"<div class='work-card'><b>准入措施</b><br>{admitted}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='work-card'><b>排除措施及原因</b><br>{excluded}</div>", unsafe_allow_html=True)
    if detail:
        unit = GovernanceUnit.from_record(row.to_dict()); before = evaluate_unit(unit, (), make_config())
        with tabs[4]:
            st.markdown(f"<div class='work-card'><b>所属关键风险路径</b><br>{row['key_paths']} 条关键路径</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='work-card'><b>当前方案配置</b><br>{detail['measures']}</div>", unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            k1.metric("治理前/后稳定系数", f"{before['fs_before_rain']:.2f} / {detail['fs_rain']:.2f}")
            k2.metric("治理前/后安全得分", f"{before['safety']:.1f} / {detail['safety']:.1f}")
            k1.metric("残余危险性 / 风险", f"{detail['residual_hazard']}级 / {detail['residual_risk']}级")
            k2.metric("全寿命周期成本", f"{detail['lifecycle_cost']:.1f} 万元")
            st.markdown(f"<div class='work-card'><b>下游转移复核</b><br>{detail['downstream_transfer_review']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='work-card'><b>专家复核事项</b><br>{detail['expert_review']}</div>", unsafe_allow_html=True)
    else:
        with tabs[4]:
            st.markdown(f"<div class='work-card'><b>所属关键风险路径</b><br>{row['key_paths']} 条关键路径</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='work-card'><b>参数依据状态</b><br>{row['evidence_level']}</div>", unsafe_allow_html=True)


def linked_unit_table() -> None:
    display = chinese_frame(st.session_state.units_df[["unit_id", "unit_name", "risk", "haz", "slope_deg", "height_m", "distance_m", "evidence_level"]])
    selected = st.session_state.selected_unit
    ids = st.session_state.units_df.unit_id.astype(str).tolist()
    order = [ids.index(selected)] + [i for i, uid in enumerate(ids) if uid != selected]
    shown = display.iloc[order].reset_index(drop=True)
    event = st.dataframe(shown, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="unit_link_table")
    rows = event.selection.rows if event and hasattr(event, "selection") else []
    if rows:
        clicked = str(shown.iloc[int(rows[0])]["治理单元编号"])
        if clicked != selected:
            st.session_state.selected_unit = clicked; st.session_state.pending_unit_selector = clicked; st.rerun()
    st.caption("当前选中单元始终置于表格首行；点击其他行可反向定位并高亮地图。")


def run_model() -> None:
    units, measures = model_inputs(); config = make_config(); holder = st.empty()
    try:
        errors = validate_project_data(st.session_state.units_df, st.session_state.measures_df, st.session_state.units_geojson, st.session_state.topology)
        if errors:
            raise ValueError("；".join(errors))
        exact = nsga = None
        verify = bool(st.session_state.verify_current_run)
        if st.session_state.solver_method == "exact" or verify:
            holder.info("正在逐一评价完整组合空间并筛选四目标非支配方案……")
            exact = solve_exact(units, measures, config, DATA_DIR / "assumptions.csv", topology=st.session_state.topology)
        if st.session_state.solver_method == "nsga2" or verify:
            bar = st.progress(0.0, text="正在初始化离散方案种群……")
            def callback(item: dict[str, object]) -> None:
                ratio = float(item["generation"]) / max(config.nsga_generations, 1)
                bar.progress(ratio, text=f"真实迭代 {item['generation']}/{config.nsga_generations}｜可行个体{item['current_feasible']}｜累计评价{item['unique_evaluations']}个唯一方案")
            nsga = solve_nsga2(units, measures, config, DATA_DIR / "assumptions.csv", topology=st.session_state.topology, progress_callback=callback)
            bar.empty()
        result = exact if st.session_state.solver_method == "exact" else nsga
        st.session_state.exact_result = exact; st.session_state.nsga_result = nsga; st.session_state.result = result
        st.session_state.validation = validate_runs(exact, [nsga]) if exact is not None and nsga is not None else None
        st.session_state.result_valid = True; st.session_state.last_optimization_hash = current_optimization_hash(); st.session_state.last_preference_hash = current_preference_hash()
        st.session_state.result_invalid_reason = ""
        st.session_state.completed_steps.add(4); st.session_state.scenario_history.append(result.run_metadata)
        holder.success(f"计算完成：{result.run_metadata['feasible_plans']:,}个可行方案，{result.run_metadata['pareto_plans']:,}个Pareto方案。")
    except Exception as exc:
        invalidate_results(str(exc)); holder.error(f"计算未完成：{exc}")


def run_multiseed_validation() -> None:
    units, measures = model_inputs(); base = make_config(); holder = st.empty()
    holder.info("正在运行精确基准与3个随机种子的工程案例验证……")
    exact = solve_exact(units, measures, base, DATA_DIR / "assumptions.csv")
    runs = []
    for seed in (42, 123, 2026):
        cfg = copy.deepcopy(base); cfg.random_seed = seed
        runs.append(solve_nsga2(units, measures, cfg, DATA_DIR / "assumptions.csv"))
    st.session_state.exact_result = exact; st.session_state.nsga_result = runs[0]; st.session_state.validation = validate_runs(exact, runs)
    holder.success("3个随机种子均已实际运行，验证指标已更新。")


def run_standard_validation() -> None:
    holder = st.empty()
    objectives_count = int(st.session_state.benchmark_objectives)
    function_name = str(st.session_state.benchmark_function)
    variables_count = int(st.session_state.benchmark_variables)
    population_size = int(st.session_state.benchmark_population)
    generations = int(st.session_state.benchmark_generations)
    mode = st.session_state.benchmark_mode
    if mode == "single":
        seeds = (int(st.session_state.benchmark_seed),)
        functions = (function_name,)
    elif mode == "multi_seed":
        seeds = tuple(int(item.strip()) for item in str(st.session_state.benchmark_batch_seeds).split(",") if item.strip())
        functions = (function_name,)
    else:
        seeds = tuple(int(item.strip()) for item in str(st.session_state.benchmark_batch_seeds).split(",") if item.strip())
        functions = tuple(str(item) for item in st.session_state.benchmark_batch_functions)
    holder.info("正在运行标准测试函数验证……")
    frame, examples = validate_standard_functions(
        functions=functions,
        seeds=seeds,
        population_size=population_size,
        generations=generations,
        objectives_count=objectives_count,
        variables_count=variables_count,
    )
    st.session_state.benchmark_results = {"summary": frame, "examples": examples}
    holder.success(f"已完成 {len(functions)} 个标准函数、{len(seeds)} 个随机种子的验证。")


def build_html_report(result) -> bytes:
    plan_id, details = selected_plan(result)
    metrics = coordination_metrics(st.session_state.topology, details) if details else {}
    selected_row = result.representatives[result.representatives.plan_id.astype(str) == plan_id].iloc[0] if plan_id else None
    summary = "尚无代表方案" if selected_row is None else f"推荐{st.session_state.selected_plan_label}（{plan_id}），估算全寿命成本{selected_row['total_cost']:.1f}万元，区域安全性{selected_row['safety']:.1f}分。"
    detail_rows = "".join(f"<tr><td>{d['unit_id']}</td><td>{d['unit_name']}</td><td>{d['measures']}</td><td>{d['safety']:.1f}</td><td>{d['lifecycle_cost']:.1f}</td></tr>" for d in details)
    html = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>科研分析报告</title><style>body{{font-family:Microsoft YaHei,Arial;max-width:1000px;margin:36px auto;color:#22384A}}h1{{color:#102A43}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #CBD7E0;padding:8px;text-align:left}}th{{background:#EAF2F6}}.note{{background:#FFF8E8;border-left:4px solid #D97706;padding:10px}}</style><body><h1>村庄地质灾害综合整治科研分析报告</h1><p>项目：{st.session_state.project_name}｜村庄：{st.session_state.village_name}｜数据版本：{st.session_state.data_version}</p><div class='note'>当前空间单元和风险路径为平台演示数据，不代表现场实测边界；成本为方案比选阶段估算，不等同于施工预算。</div><h2>推荐结论</h2><p>{summary}</p><h2>单元措施配置</h2><table><tr><th>编号</th><th>单元</th><th>措施</th><th>安全性</th><th>估算成本（万元）</th></tr>{detail_rows}</table><h2>空间协同</h2><p>{'；'.join(f'{k}：{v}' for k,v in metrics.items())}</p><h2>模型边界</h2><p>本报告用于科研研判和方案比选，参数仍需结合规范、现场调查、工程案例与专家评审标定。</p></body></html>"""
    return html.encode("utf-8")


def export_zip(result) -> bytes:
    items = {"01_治理单元与标签.csv": chinese_frame(result.tagged_units), "02_工程措施库.csv": chinese_frame(result.measure_library),
             "03_准入筛选结果.csv": chinese_frame(result.screening), "04_可行方案.csv": chinese_frame(result.all_feasible),
             "05_Pareto方案.csv": chinese_frame(result.pareto), "06_代表方案.csv": chinese_frame(result.representatives),
             "07_代表方案单元明细.csv": chinese_frame(result.representative_details), "08_参数依据与复核清单.csv": parameters_and_assumptions(result)}
    if not result.iteration_history.empty:
        items["09_NSGA-II迭代历史.csv"] = chinese_frame(result.iteration_history)
    if st.session_state.validation is not None:
        items["10_工程算法验证指标.csv"] = st.session_state.validation.metrics; items["11_多随机种子汇总.csv"] = st.session_state.validation.seed_summary
    if st.session_state.benchmark_results is not None:
        items["12_标准函数验证汇总.csv"] = st.session_state.benchmark_results["summary"]
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in items.items(): archive.writestr(name, csv_bytes(frame))
        archive.writestr("13_完整运行结果.json", result_to_json(result).encode("utf-8"))
        archive.writestr("14_科研分析报告.html", build_html_report(result))
        archive.writestr("15_字段中英文映射.json", json.dumps(FIELD_LABELS, ensure_ascii=False, indent=2).encode("utf-8"))
    return buffer.getvalue()


def header() -> None:
    result = st.session_state.result if st.session_state.result_valid else None
    scenario = result.run_metadata.get("scenario_id", "尚未计算") if result else "尚未计算"
    status = "结果有效" if result else ("结果已失效" if st.session_state.result is not None else "待计算")
    left, right = st.columns([3.7, 2])
    with left:
        st.markdown("<div class='platform-head'><h1>综合整治工程技术多目标优化智能研判与决策平台</h1><div class='platform-sub'>治理单元识别 · 多维综合评估 · 多目标优化求解 · 方案评审与成果输出</div></div>", unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='platform-head meta-line'>项目：{st.session_state.project_name}<br>村庄：{st.session_state.village_name}｜场景：{scenario}<br>数据：{st.session_state.data_version}｜{status}</div>", unsafe_allow_html=True)
    st.markdown("<div class='notice'><b>数据边界：</b>当前空间单元和风险路径为平台演示数据，不代表现场实测边界；估算成本用于方案比选，不等同于施工预算。</div>", unsafe_allow_html=True)
    if st.session_state.result is not None and not st.session_state.result_valid:
        reason = st.session_state.result_invalid_reason or "当前输入条件已发生变化，现有结果已经失效，请重新运行优化计算。"
        st.markdown(f"<div class='invalid'>{reason}</div>", unsafe_allow_html=True)


def sidebar() -> tuple[bool, bool, bool]:
    quick = run_clicked = validate_clicked = False
    with st.sidebar:
        st.markdown("## 工作模式")
        mode_options = ["平台首页", "引导式研判", "快速分析", "算法验证中心"]
        mode_choice = st.radio("当前模式", mode_options, index=mode_options.index(st.session_state.mode), horizontal=True, label_visibility="collapsed")
        if mode_choice != st.session_state.mode:
            switch_mode(mode_choice)
        st.markdown("## 项目摘要")
        st.caption(f"{st.session_state.project_name}｜{st.session_state.village_name}")
        st.caption(f"数据版本：{st.session_state.data_version}")
        st.caption(f"数据来源：{st.session_state.data_source_label}")
        st.caption(f"参数状态：{st.session_state.scenario_parameter_source}")
        if is_guided_mode():
            step = int(st.session_state.current_step)
            st.markdown("## 流程导航")
            st.markdown(f"<div class='section-label'>{STEP_STAGE_TITLE[step]}</div>", unsafe_allow_html=True)
            progress_lines = []
            for i, label in enumerate(GUIDED_STEPS):
                cls = "step-now" if i == step else ("step-done" if i in st.session_state.completed_steps else "")
                marker = "●" if i == step else ("✓" if i in st.session_state.completed_steps else "○")
                progress_lines.append(f"<div class='{cls}'>{marker} {i+1}. {label}</div>")
            st.markdown("<div class='step-list'>" + "".join(progress_lines) + "</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='task-card'><b>本步目标</b><br>{STEP_OUTPUTS[step]}</div>", unsafe_allow_html=True)
            stage_id = current_stage_id()
            stage_labels = [STAGE_LABELS[item] for item in STAGE_IDS]
            phase_choice = st.radio("跳转阶段", stage_labels, index=STAGE_IDS.index(stage_id), label_visibility="collapsed")
            selected_stage_id = STAGE_IDS[stage_labels.index(phase_choice)]
            if selected_stage_id != stage_id:
                go_to_guided_step(STAGE_TO_STEPS[selected_stage_id][0])
            step_choices = list(STAGE_TO_STEPS[stage_id])
            step_labels = [f"{item + 1}. {GUIDED_STEPS[item]}" for item in step_choices]
            step_choice = st.selectbox("阶段内子步骤", step_labels, index=step_choices.index(step), label_visibility="visible")
            selected_step = step_choices[step_labels.index(step_choice)]
            if selected_step != step:
                go_to_guided_step(selected_step)
            if step == 0:
                st.caption("当前步骤只校验项目、研究区和基础资料状态，不提前展开治理单元结论。")
            elif step == 7:
                st.caption("完成专家复核后导出成果。")
        elif is_home_mode():
            st.markdown("<div class='task-card'><b>平台首页</b><br>用于查看研究任务、措施库概览、四阶段流程和当前数据状态，并进入快速分析或引导式研判。</div>", unsafe_allow_html=True)
        elif is_validation_mode():
            st.markdown("<div class='task-card'><b>算法验证中心</b><br>用于查看基础工程案例与标准测试函数验证，不影响普通业务用户完成正式研判流程。</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='task-card'><b>快速分析</b><br>单页完成输入、约束、求解和结果输出，不显示八步骤导航。</div>", unsafe_allow_html=True)
            ready, blockers = quick_analysis_requirements()
            if blockers:
                st.caption("当前不可直接计算：")
                for item in blockers[:4]:
                    st.write(f"- {item}")
        st.divider()
        if st.button("恢复示例场景", use_container_width=True):
            restore_example_scenario()
    return quick, run_clicked, validate_clicked


def solver_controls(compact: bool = False, key_map: dict[str, str] | None = None) -> None:
    key_map = key_map or {}
    st.selectbox("求解方法", ["exact", "nsga2"], format_func=lambda x: "精确枚举" if x == "exact" else "NSGA-II智能搜索", key=key_map.get("solver_method", "solver_method"))
    st.number_input(f"总预算（{st.session_state.budget_unit}）", 0.0, 100000.0, step=10.0, key=key_map.get("total_budget", "total_budget"), help="约束全村各单元的估算全寿命周期成本总和。")
    st.number_input("治理后稳定系数下限", .5, 2.0, step=.01, key=key_map.get("min_fs", "min_fs"))
    st.number_input(f"最低安全得分（{st.session_state.safety_unit}）", 0.0, 100.0, step=1.0, key=key_map.get("min_safety", "min_safety"), help="综合稳定性和灾害链控制能力形成的0—100分指标；默认值来自示例场景配置，仍需项目标定。")
    if not compact:
        with st.expander("组合与作用参数"):
            st.selectbox("每单元最多措施数", [1, 2], key=key_map.get("max_measures", "max_measures")); st.checkbox("允许中高风险单元不治理", key=key_map.get("allow_untreated", "allow_untreated"))
            st.slider("排水组合协同系数", 1.0, 1.3, step=.01, key="drainage_synergy"); st.slider("第二项措施效果折减", .5, 1.0, step=.01, key="second_factor")
    solver_method = st.session_state[key_map.get("solver_method", "solver_method")]
    if solver_method == "nsga2":
        with st.expander("NSGA-II搜索参数", expanded=True):
            st.number_input("种群规模", 20, 1000, step=10, key="population_size"); st.number_input("最大迭代代数", 1, 1000, step=5, key="generations")
            st.slider("交叉概率", 0.0, 1.0, step=.05, key="crossover_probability"); st.slider("变异概率", 0.0, 1.0, step=.05, key="mutation_probability")
            st.number_input("随机种子", 0, 999999, step=1, key="random_seed")
    if not compact:
        st.checkbox("同时运行精确基准进行本次对照", key="verify_current_run", help="仅在需要算法验证时启用；快速分析默认关闭。")


def preference_controls() -> None:
    st.caption("偏好权重只改变Pareto集内的代表方案选择，不重新运行优化。")
    st.slider("安全偏好", 0.0, 1.0, step=.05, key="weight_safety"); st.slider("经济偏好", 0.0, 1.0, step=.05, key="weight_economy")
    st.slider("可持续偏好", 0.0, 1.0, step=.05, key="weight_sustainability"); st.slider("韧性偏好", 0.0, 1.0, step=.05, key="weight_resilience")


def map_controls() -> None:
    st.selectbox("底图", ["卫星影像", "卫星影像+标注", "街道地图", "地形图", "离线坐标网格"], key="map_background")
    st.selectbox("地图表达", ["治理前风险", "治理后安全性", "推荐措施", "约束未满足单元"], key="display_mode")
    st.checkbox("显示关键风险路径", key="show_paths")


def navigation() -> None:
    apply_page_state(str(st.session_state.current_page_id))


def render_home_page() -> None:
    summary = current_data_summary()
    st.markdown("### 平台首页")
    st.markdown(
        "<div class='work-card'><b>平台说明</b><br>"
        "首页用于说明综合整治工程技术比选与决策平台的研究任务、正式四阶段计算框架、当前数据状态，以及进入“快速分析”或“引导式研判”的统一入口。"
        "</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.25, 1.75])
    with c1:
        st.markdown("#### 当前项目与数据状态")
        cols = st.columns(2)
        cols[0].metric("项目名称", summary["project_name"] or "未填写")
        cols[1].metric("研究区", summary["village_name"] or "未填写")
        cols[0].metric("数据版本", summary["data_version"] or "未填写")
        cols[1].metric("数据来源", summary["data_source"])
        cols[0].metric("治理单元", summary["unit_count"])
        cols[1].metric("风险路径", summary["path_count"])
        cols[0].metric("字段完整率", f"{summary['field_completeness']:.1f}%")
        cols[1].metric("措施数量", summary["measure_count"])
        st.markdown(f"<div class='work-card'><b>空间范围</b><br>{summary['spatial_bounds']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='work-card'><b>参数依据状态</b><br>{summary['parameter_status']}</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        if a.button("进入快速分析", type="primary", use_container_width=True):
            switch_mode("快速分析")
        if b.button("进入引导式研判", use_container_width=True):
            switch_mode("引导式研判")
        if st.button("进入算法验证中心", use_container_width=True):
            switch_mode("算法验证中心")
    with c2:
        render_map([], "home_overview_map", show_units=False)
        st.caption("首页地图仅展示研究区基础底图和空间资料状态，不提前展开正式治理单元识别结果、准入结论或 Pareto 方案。")
    st.markdown("#### 四阶段正式计算框架")
    stage_cols = st.columns(4)
    for column, (title, detail) in zip(stage_cols, HOME_STAGE_SUMMARY, strict=True):
        column.markdown(f"<div class='work-card'><b>{title}</b><br>{detail}</div>", unsafe_allow_html=True)
    st.markdown("#### 工程措施库概览")
    measure_cols = st.columns(2)
    for index, item in enumerate(FORMAL_MEASURE_LIBRARY):
        measure_cols[index % 2].markdown(
            f"<div class='work-card'><b>{item['name']}</b><br>"
            f"功能：{item['function']}<br>"
            f"适用对象：{item['target']}<br>"
            f"主要限制：{item['limit']}<br>"
            f"依据状态：{item['status']}</div>",
            unsafe_allow_html=True,
        )


def render_project_overview(result) -> None:
    left, right = st.columns([3, 1.15])
    with left:
        render_map([], "overview_map", show_units=False)
        _, _, _, attribution = _map_style_and_layers(st.session_state.map_background)
        st.caption(f"默认底图：{st.session_state.map_background}；底图署名：{attribution}。第一步只展示基础资料，不提前展开治理单元与风险路径结论。")
    with right:
        st.markdown("### 第一步前置检查")
        a, b = st.columns(2); a.metric("研究区图层", len(st.session_state.context_geojson.get("features", []))); b.metric("数据状态", "完整" if not st.session_state.result_invalid_reason else "待复核")
        c, d = st.columns(2); c.metric("原始资料状态", "已加载"); d.metric("后续治理单元数据", f"{len(st.session_state.units_df)} 条（仅状态）")
        st.markdown("<div class='work-card'><b>当前仅展示</b><br>项目名称、研究区名称、基础底图、村庄/道路/水系/承灾体图层与资料完整性概况。</div>", unsafe_allow_html=True)
        st.markdown("<div class='work-card'><b>本步不提前展示</b><br>正式治理单元编号、工程准入结论、风险路径覆盖结果、Pareto方案与治理前后评分。</div>", unsafe_allow_html=True)


def render_data_units(result) -> None:
    st.markdown("### 治理单元导入、识别与复核")
    st.caption("本步区分原始调查资料、治理单元识别建议和已确认治理单元；通过校验后方可进入准入筛选。")
    _, details = selected_plan(result)
    map_controls_inline = st.columns([1, 1, 1, 3])
    with map_controls_inline[0]: st.selectbox("底图类型", ["卫星影像", "卫星影像+标注", "街道地图", "地形图", "离线坐标网格"], key="map_background")
    with map_controls_inline[1]: st.selectbox("表达模式", ["治理前风险", "治理后安全性", "推荐措施", "约束未满足单元"], key="display_mode")
    with map_controls_inline[2]: st.checkbox("风险路径", key="show_paths")
    left, right = st.columns([2.8, 1.25])
    with left: render_map(details, "units_map")
    with right: unit_detail(result, details)
    linked_unit_table()
    with st.expander("编辑或导入项目数据"):
        st.info("支持CSV、Excel和GeoJSON/JSON。导入后会校验必填字段、类型、编号唯一性、GeoJSON对应关系和风险路径引用。")
        unit_file = st.file_uploader("治理单元表", type=["csv", "xlsx", "xls"], key="unit_upload")
        measure_file = st.file_uploader("措施库表", type=["csv", "xlsx", "xls"], key="measure_upload")
        geo_file = st.file_uploader("治理单元GeoJSON", type=["geojson", "json"], key="geo_upload")
        topo_file = st.file_uploader("风险路径JSON", type=["json"], key="topo_upload")
        t1, t2, t3 = st.columns(3)
        t1.download_button("下载治理单元模板CSV", csv_bytes(st.session_state.units_df.head(0)), file_name="治理单元模板.csv", mime="text/csv", use_container_width=True)
        t2.download_button("下载措施库模板CSV", csv_bytes(st.session_state.measures_df.head(0)), file_name="措施库模板.csv", mime="text/csv", use_container_width=True)
        t3.download_button("下载当前拓扑示例JSON", json.dumps(st.session_state.topology, ensure_ascii=False, indent=2).encode("utf-8"), file_name="风险路径示例.json", mime="application/json", use_container_width=True)
        if st.button("校验并载入所选数据"):
            try:
                units_df = read_uploaded_table(unit_file) if unit_file else st.session_state.units_df
                measures_df = read_uploaded_table(measure_file) if measure_file else st.session_state.measures_df
                geojson = json.load(geo_file) if geo_file else st.session_state.units_geojson
                topology = json.load(topo_file) if topo_file else st.session_state.topology
                errors = validate_project_data(units_df, measures_df, geojson, topology)
                if errors: st.error("；".join(errors))
                else:
                    apply_loaded_data(
                        units_df,
                        measures_df,
                        geojson,
                        topology,
                        source_label="引导式研判已确认治理单元数据",
                        source_kind="guided",
                    )
                    st.success("数据通过一致性校验并已载入。")
            except Exception as exc: st.error(f"导入失败：{exc}")
        st.session_state.units_df = st.data_editor(st.session_state.units_df, key=f"units_editor_{st.session_state.editor_version}", use_container_width=True, num_rows="dynamic", column_config=editor_columns(st.session_state.units_df))


def render_screening() -> None:
    units, measures = model_inputs(); screening = pd.DataFrame([item.to_dict() for item in screen_all(units, measures)])
    st.markdown("### 工程措施准入判定")
    st.info(f"{len(screening)}条判定记录＝{len(units)}个治理单元×{len(measures)}类候选措施，并不表示{len(screening)}种不同工程措施。M01—M08是措施类别编码；每条判定使用Uxx-Mxx唯一编号。")
    a, b, c = st.columns(3); a.metric("判定记录", len(screening)); b.metric("准入", int(screening.eligible.sum())); c.metric("剔除", int((~screening.eligible).sum()))
    render_frame(chinese_frame(screening), use_container_width=True, hide_index=True)
    with st.expander("准入条件与符号说明"):
        st.markdown("- 汇流面积 *Ac*，单位为平方千米。例如“Ac<0.05 km²”表示汇流面积小于0.05平方千米的沟谷或小型汇水区。\n- 坡度 *θ*，单位为度；坡高 *H*、坡脚空间 *W*，单位为米。\n- 风险与危险性均采用1—5级，数值越高表示关注程度越高。每条记录同时给出命中条件、限制条件、准入结论和剔除原因。")


def render_multidimensional_evaluation() -> None:
    st.markdown("### 多维综合评估配置")
    st.caption("本页用于核对措施适用性、四维评价参数、地形地貌与工程适宜性修正、以及区域协同关系配置；真正的村域四目标仍在候选方案评价阶段通过统一评价函数计算。")
    units, measures = model_inputs()
    screening = pd.DataFrame([item.to_dict() for item in screen_all(units, measures)])
    tabs = st.tabs(["适用性初筛摘要", "四维评价与适宜性修正", "区域协同关系"])
    with tabs[0]:
        cols = st.columns(4)
        cols[0].metric("治理单元", len(units))
        cols[1].metric("措施类型", len(measures))
        cols[2].metric("准入记录", len(screening))
        cols[3].metric("准入比例", f"{(screening.eligible.mean() * 100) if not screening.empty else 0:.1f}%")
        render_frame(
            chinese_frame(
                screening[["screening_id", "unit_id", "measure_name", "eligible", "exclusion_reason", "evidence_level"]]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with tabs[1]:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("<div class='work-card'><b>四维评价口径</b><br>安全性、经济性、可持续性和韧性均采用 0—100 分、数值越大越优的展示口径；若求解器内部采用最小化形式，则统一转换为 100－得分。</div>", unsafe_allow_html=True)
            st.markdown("<div class='work-card'><b>经济性双轨口径</b><br>LCI/SC 用于经济性目标展示与 Pareto 比较；货币化全寿命周期总成本用于预算硬约束、金额展示和成果导出，不作为第二套经济目标。</div>", unsafe_allow_html=True)
        with right:
            c1, c2 = st.columns(2)
            c1.metric("稳定系数下限", f"{st.session_state.min_fs:.2f}")
            c2.metric("最低安全得分", f"{st.session_state.min_safety:.0f} {st.session_state.safety_unit}")
            c1.metric("排水协同系数", f"{st.session_state.drainage_synergy:.2f}")
            c2.metric("第二措施折减", f"{st.session_state.second_factor:.2f}")
            assumptions = pd.read_csv(DATA_DIR / "assumptions.csv")
            render_frame(chinese_frame(assumptions.head(12)), use_container_width=True, hide_index=True)
    with tabs[2]:
        top_left, top_right = st.columns([1.3, 1])
        with top_left:
            st.markdown("<div class='work-card'><b>区域协同正式位置</b><br>区域协同关系在本阶段完成配置，并在阶段三每一个候选方案评价时进入统一评价函数，形成村域四目标值后再参与 Pareto 搜索。</div>", unsafe_allow_html=True)
            st.markdown("<div class='work-card'><b>统一数据源</b><br>地图路径展示、关键路径集合、路径覆盖率、韧性评价和区域协同分析必须读取同一套 topology / 路径数据。</div>", unsafe_allow_html=True)
        with top_right:
            t1, t2 = st.columns(2)
            t1.metric("关键路径数量", len(st.session_state.topology.get("paths", [])))
            t2.metric("当前数据来源", st.session_state.data_source_label)
            t1.metric("空间范围", spatial_bounds_summary())
            t2.metric("参数状态", st.session_state.scenario_parameter_source)
        render_frame(pd.DataFrame(st.session_state.topology.get("paths", [])), use_container_width=True, hide_index=True)


def render_solver(result) -> None:
    st.markdown("### 多目标优化智能研判")
    info, config_area = st.columns([1.5, 1])
    with info:
        st.markdown("<div class='work-card'><b>精确枚举</b><br>逐一计算全部可能方案；当前4单元小案例优先使用，可得到完整基准解。组合数量增大后计算成本快速上升。</div>", unsafe_allow_html=True)
        st.markdown("<div class='work-card'><b>NSGA-II智能搜索</b><br>不遍历全部方案，通过种群、选择、交叉与变异近似搜索Pareto方案；适用于更大组合空间，结果需要验证稳定性和覆盖质量。</div>", unsafe_allow_html=True)
        st.markdown("<div class='notice'><b>安全底线解释：</b>稳定系数下限直接约束降雨工况的抗滑稳定；安全得分还包含灾害链控制能力。二者分别防止“力学不安全”和“综合得分掩盖单项短板”，因此同时保留。默认58分是示例阈值，需结合规范、工程案例和专家评审标定。</div>", unsafe_allow_html=True)
    with config_area:
        st.markdown("#### 当前配置")
        st.write(f"求解方法：{'精确枚举' if st.session_state.solver_method == 'exact' else 'NSGA-II智能搜索'}")
        st.write(f"预算：{st.session_state.total_budget:.0f}万元｜稳定系数下限：{st.session_state.min_fs:.2f}｜安全得分下限：{st.session_state.min_safety:.0f}")
        st.caption("说明：区域协同在每个候选方案的后台评价链中已经参与四目标形成；本页只是触发统一计算。")
        if st.button("立即运行优化计算", type="primary", use_container_width=True): run_model()
        if st.button("进入算法验证中心", use_container_width=True):
            switch_mode("算法验证中心")
    if result:
        meta = result.run_metadata; cols = st.columns(4)
        cols[0].metric("搜索空间", f"{meta['estimated_combinations']:,}"); cols[1].metric("实际评价", f"{meta.get('unique_evaluations', meta['estimated_combinations']):,}")
        cols[2].metric("可行方案", f"{meta['feasible_plans']:,}"); cols[3].metric("Pareto方案", f"{meta['pareto_plans']:,}")
        if not result.iteration_history.empty:
            hist = result.iteration_history
            fig = px.line(hist, x="generation", y="archive_hypervolume", markers=True, labels={"generation": "迭代代数", "archive_hypervolume": "累计档案归一化超体积"})
            fig.update_layout(title="NSGA-II累计非支配档案覆盖趋势", height=320); st.plotly_chart(fig, use_container_width=True)
            st.caption("图表解读：超体积依据累计非支配档案计算，整体不下降时表示目标空间覆盖持续改善；它不再被表述为到人为理想点的“收敛距离”。")


def render_algorithm_validation() -> None:
    tabs = st.tabs(["标准测试函数", "村庄工程案例"])
    with tabs[0]:
        st.markdown("### 标准测试函数验证工作台")
        st.info("ZDT 和 DTLZ 用于验证 NSGA-II 连续变量机制、非支配排序、拥挤距离、精英保留和解集分布；不能替代村庄工程离散编码与正式工程约束验证。")
        spec_frame = benchmark_spec_table()
        render_frame(spec_frame, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.selectbox("目标函数个数", [2, 3, 4], key="benchmark_objectives")
        function_options = benchmark_functions_for_objectives(int(st.session_state.benchmark_objectives))
        if st.session_state.benchmark_function not in function_options:
            st.session_state.benchmark_function = function_options[0]
        c2.selectbox("标准测试函数", function_options, key="benchmark_function")
        default_variables = BENCHMARK_SPECS[st.session_state.benchmark_function].suggested_variables
        c3.number_input("决策变量维度", min_value=max(2, int(st.session_state.benchmark_objectives)), value=int(default_variables), step=1, key="benchmark_variables")
        c4, c5, c6 = st.columns(3)
        c4.number_input("种群规模", 20, 500, step=10, key="benchmark_population")
        c5.number_input("最大代数", 10, 500, step=10, key="benchmark_generations")
        c6.selectbox("运行方式", ["single", "multi_seed", "multi_function"], format_func=lambda x: {"single": "单次运行", "multi_seed": "多随机种子", "multi_function": "多函数批量"}[x], key="benchmark_mode")
        if st.session_state.benchmark_mode == "single":
            st.number_input("随机种子", 0, 999999, step=1, key="benchmark_seed")
        else:
            st.text_input("随机种子列表（逗号分隔）", key="benchmark_batch_seeds")
        if st.session_state.benchmark_mode == "multi_function":
            st.multiselect("批量函数", function_options, default=function_options[: min(3, len(function_options))], key="benchmark_batch_functions")
        spec = BENCHMARK_SPECS[st.session_state.benchmark_function]
        st.markdown(
            f"<div class='work-card'><b>{spec.name}</b><br>"
            f"支持目标数：{'、'.join(str(item) for item in spec.objectives)}<br>"
            f"建议维度：{spec.suggested_variables}<br>"
            f"变量范围：{spec.variable_range}<br>"
            f"Pareto前沿特征：{spec.front_description}<br>"
            f"适合检验：{spec.capability}</div>",
            unsafe_allow_html=True,
        )
        if st.button("开始标准函数验证", type="primary"): run_standard_validation()
        bundle = st.session_state.benchmark_results
        if bundle:
            render_frame(bundle["summary"], use_container_width=True, hide_index=True)
            name = st.selectbox("查看函数", list(bundle["examples"]))
            example = bundle["examples"][name]
            if example.objectives.shape[1] == 2:
                fig = go.Figure(); fig.add_trace(go.Scatter(x=example.theoretical_front[:, 0], y=example.theoretical_front[:, 1], mode="lines", name="理论Pareto前沿"))
                fig.add_trace(go.Scatter(x=example.objectives[:, 0], y=example.objectives[:, 1], mode="markers", name="算法解集")); fig.update_layout(title=f"{name}理论前沿与算法解集", xaxis_title="目标1", yaxis_title="目标2")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter_3d(x=example.objectives[:, 0], y=example.objectives[:, 1], z=example.objectives[:, 2], title=f"{name}非支配解分布（三维投影）")
                st.plotly_chart(fig, use_container_width=True)
            fig = px.line(example.history, x="迭代代数", y="IGD", title=f"{name}收敛过程（IGD）")
            st.plotly_chart(fig, use_container_width=True)
            if "Hypervolume" in example.history:
                hv_fig = px.line(example.history, x="迭代代数", y="Hypervolume", title=f"{name}超体积变化")
                st.plotly_chart(hv_fig, use_container_width=True)
            st.caption("图表解读：IGD 用于表示算法解集与参考 Pareto 前沿的距离；Hypervolume 用于综合反映收敛与覆盖；Spacing 用于描述分布均匀性。")
    with tabs[1]:
        st.markdown("### 第二层：村庄工程离散案例验证")
        st.info("精确枚举作为小案例完整基准，用于检验离散编码、准入/互斥、硬约束、修复和工程评价耦合是否正确。")
        if st.button("运行精确基准＋3随机种子NSGA-II", type="primary"): run_multiseed_validation()
        if st.session_state.validation:
            render_frame(st.session_state.validation.metrics, use_container_width=True, hide_index=True)
            render_frame(st.session_state.validation.seed_summary, use_container_width=True, hide_index=True)
            st.markdown(f"<div class='ok'>{st.session_state.validation.conclusion}</div>", unsafe_allow_html=True)


def render_plans(result) -> None:
    if not result:
        st.warning("请先完成优化计算，再进入方案比选。")
        return
    st.markdown("### Pareto候选方案与唯一代表方案")
    preference_controls()
    rebuild_representatives(result)
    st.selectbox("当前联动代表方案", result.representatives.recommendation_type.tolist(), key="selected_plan_label")
    projections = {"成本—安全性": ("total_cost", "safety"), "经济性—安全性": ("economy", "safety"), "可持续性—韧性": ("sustainability", "resilience")}
    projection = st.selectbox("二维投影", list(projections)); x, y = projections[projection]
    fig = px.scatter(result.pareto, x=x, y=y, hover_data=["plan_id", "configuration_summary"], labels={"total_cost": "估算全寿命成本（万元）", "safety": "安全性得分", "economy": "经济性得分", "sustainability": "可持续性得分", "resilience": "韧性得分"})
    fig.add_trace(go.Scatter(x=result.representatives[x], y=result.representatives[y], mode="markers+text", text=result.representatives.recommendation_type, textposition="top center", marker={"size": 13, "symbol": "diamond", "color": COLORS["orange"]}, name="代表方案"))
    fig.update_layout(title=f"四目标Pareto解集在{projection}平面上的二维投影", height=440); st.plotly_chart(fig, use_container_width=True)
    st.caption("图表解读：Pareto判定同时使用安全性、经济性、可持续性与韧性；当前图只是选择其中两个指标进行二维投影。菱形标记为五类代表方案。")
    heat = result.representatives.set_index("recommendation_type")[["safety", "economy", "sustainability", "resilience"]]
    fig = px.imshow(heat, text_auto=".1f", aspect="auto", color_continuous_scale=[[0, "#E8EEF3"], [1, COLORS["cyan"]]], zmin=0, zmax=100, labels={"x": "评价目标", "y": "代表方案", "color": "得分"})
    fig.update_layout(title="代表方案四目标标准化热力图", height=350); st.plotly_chart(fig, use_container_width=True)
    st.caption("图表解读：热力图用于比较不同代表方案的优势方向，避免五条雷达曲线相互遮挡；最终推荐仍需结合成本增量和空间薄弱环节复核。")
    plan_id, details = selected_plan(result); selected = result.representatives[result.representatives.plan_id.astype(str) == plan_id].iloc[0]
    cheapest = result.representatives.sort_values("total_cost").iloc[0]
    metrics = coordination_metrics(st.session_state.topology, details)
    tie_count = int(selected.get("tie_candidate_count", 1))
    tie_note = f"主目标并列候选共{tie_count}个；本次按{selected.get('tie_break_rule', '既定规则')}确定唯一代表方案。" if tie_count > 1 else "主目标不存在并列极值，直接按主指标选出唯一代表方案。"
    st.markdown(f"<div class='work-card'><b>推荐结论：</b>{st.session_state.selected_plan_label}（{plan_id}）。估算全寿命成本{selected['total_cost']:.1f}万元；相较代表方案中的最低成本方案增加{selected['total_cost']-cheapest['total_cost']:.1f}万元，安全性提高{selected['safety']-cheapest['safety']:.1f}分。<br><b>并列处理：</b>{tie_note}<br><b>空间复核：</b>{metrics['未治理薄弱环节']}<br><b>参数复核：</b>{selected['expert_review']}<br><b>限制：</b>示例空间边界和待标定参数不能直接替代现场勘察与施工设计。</div>", unsafe_allow_html=True)
    render_frame(chinese_frame(pd.DataFrame(details)), use_container_width=True, hide_index=True)


def render_coordination(result) -> None:
    if not result:
        st.warning("请先完成优化计算，再开展空间协同研判。")
        return
    st.markdown("<div class='notice'><b>说明：</b>本页只解析和复核已经参与第6步优化计算的村域协同结果，不在此处首次计算区域协同评分。</div>", unsafe_allow_html=True)
    st.selectbox("联动代表方案", result.representatives.recommendation_type.tolist(), key="selected_plan_label")
    plan_id, details = selected_plan(result); metrics = coordination_metrics(st.session_state.topology, details)
    left, right = st.columns([2.8, 1.2])
    with left: render_map(details, "coordination_map")
    with right:
        st.markdown(f"### {plan_id}｜协同指标")
        for key in ["路径治理覆盖率（%）", "物源控制率（%）", "输移拦截率（%）", "承灾体防护率（%）", "上下游协同贡献（%）", "路径平均残余风险（%）"]:
            st.metric(key, f"{metrics[key]:.1f}%")
        st.markdown(f"<div class='notice'><b>未治理薄弱环节：</b>{metrics['未治理薄弱环节']}</div>", unsafe_allow_html=True)
    rows = []
    detail_map = {str(item["unit_id"]): item for item in details}
    for path in st.session_state.topology.get("paths", []):
        stages = path_stage_status(detail_map, path)
        rows.append({"路径编号": path["path_id"], "风险路径": path["name"], "物源控制": "有效" if stages[0] else "未覆盖", "输移拦截": "有效" if stages[1] else "未覆盖", "承灾体防护": "有效" if stages[2] else "未覆盖", "完整控制": "是" if all(stages) else "否"})
    render_frame(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("图表解读：只有措施类型与物源、输移或承灾体环节相匹配时才计为有效覆盖。本页读取与优化阶段一致的路径数据和覆盖逻辑，只负责解释与复核。")


def render_results(result) -> None:
    if not result:
        st.warning("暂无有效成果，请先运行优化计算。")
        return
    meta = result.run_metadata
    st.markdown("### 场景追踪与成果导出")
    render_frame(pd.DataFrame([{ "项目名称": st.session_state.project_name, "村庄名称": st.session_state.village_name, "数据版本": st.session_state.data_version,
        "场景编号": meta["scenario_id"], "创建时间": meta["run_time"], "配置标识": meta["configuration_hash"],
        "求解方法": "精确枚举" if meta["solver"] == "exact_enumeration" else "NSGA-II智能搜索", "随机种子": meta["random_seed"],
        "结果状态": "有效", "数据性质": "平台演示数据"}]), use_container_width=True, hide_index=True)
    st.download_button("下载当前结果JSON", result_to_json(result), file_name="当前场景完整结果.json", mime="application/json", use_container_width=True)
    st.download_button("下载科研分析报告HTML", build_html_report(result), file_name="科研分析报告.html", mime="text/html", use_container_width=True)
    st.download_button("下载完整成果包ZIP", export_zip(result), file_name="村庄综合整治研判成果.zip", mime="application/zip", use_container_width=True)
    st.markdown("<div class='notice'><b>专家复核清单：</b>措施效应与成本标定；排水出口及下游风险；抗滑桩滑面与嵌固段；示例GeoJSON替换；稳定系数和安全得分阈值复核。</div>", unsafe_allow_html=True)


def quick_source_selector() -> tuple[bool, list[str]]:
    st.selectbox("数据来源", QUICK_DATA_SOURCES, key="quick_data_source")
    source = st.session_state.quick_data_source
    if source == QUICK_DATA_SOURCES[0]:
        st.markdown("<div class='notice'><b>说明：</b>当前使用内置示例场景，仅用于开发示例与流程验收，不代表真实工程调查成果。</div>", unsafe_allow_html=True)
        if st.button("载入内置示例场景", use_container_width=True):
            restore_example_scenario()
        return quick_analysis_requirements("example")
    if source == QUICK_DATA_SOURCES[1]:
        snapshot = st.session_state.guided_snapshot
        if snapshot is None:
            blockers = ["尚未在引导式研判中完成治理单元确认，当前不能引用“已确认治理单元数据”。"]
            st.markdown("<div class='invalid'>尚未形成已确认治理单元数据，请先在阶段一完成识别、校验与确认。</div>", unsafe_allow_html=True)
            return False, blockers
        if st.button("载入已确认治理单元数据", use_container_width=True):
            request_scenario_apply({
                "units_df": normalize_units_frame(snapshot["units_df"]),
                "measures_df": normalize_measures_frame(snapshot["measures_df"]),
                "units_geojson": copy.deepcopy(snapshot["units_geojson"]),
                "topology": copy.deepcopy(snapshot["topology"]),
                "source_label": "引导式研判已确认治理单元数据",
                "source_kind": "guided",
                "project_name": snapshot.get("project_name"),
                "village_name": snapshot.get("village_name"),
                "data_version": snapshot.get("data_version"),
            })
        if st.session_state.data_source_kind == "guided":
            st.markdown("<div class='ok'>当前引用引导式研判中已确认的数据，快速分析与引导式研判共用同一套数据模型和评价链。</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='notice'><b>提示：</b>请先点击“载入已确认治理单元数据”，再开始快速分析。</div>", unsafe_allow_html=True)
        return quick_analysis_requirements("guided")
    st.markdown("<div class='work-card'><b>上传数据文件</b><br>支持 CSV、Excel、GeoJSON 和 JSON。载入后将执行与引导式研判相同的数据校验。</div>", unsafe_allow_html=True)
    revision = int(st.session_state.form_revision)
    unit_file = st.file_uploader("治理单元表", type=["csv", "xlsx", "xls"], key=f"quick_unit_upload_{revision}")
    measure_file = st.file_uploader("措施库表", type=["csv", "xlsx", "xls"], key=f"quick_measure_upload_{revision}")
    geo_file = st.file_uploader("治理单元 GeoJSON", type=["geojson", "json"], key=f"quick_geo_upload_{revision}")
    topo_file = st.file_uploader("风险路径 JSON", type=["json"], key=f"quick_topo_upload_{revision}")
    identity_cols = st.columns(3)
    quick_project_name = identity_cols[0].text_input("项目名称", key="quick_project_name")
    quick_village_name = identity_cols[1].text_input("研究区名称", key="quick_village_name")
    quick_data_version = identity_cols[2].text_input("数据版本", key="quick_data_version")
    if st.button("校验并载入上传数据", use_container_width=True):
        try:
            if not all([unit_file, measure_file, geo_file, topo_file]):
                raise ValueError("请完整提供治理单元表、措施库表、治理单元 GeoJSON 和风险路径 JSON。")
            units_df = read_uploaded_table(unit_file)
            measures_df = read_uploaded_table(measure_file)
            geojson = read_uploaded_json(geo_file)
            topology = read_uploaded_json(topo_file)
            errors = validate_project_data(units_df, measures_df, geojson, topology)
            if errors:
                raise ValueError("；".join(errors))
            request_scenario_apply({
                "units_df": normalize_units_frame(units_df),
                "measures_df": normalize_measures_frame(measures_df),
                "units_geojson": copy.deepcopy(geojson),
                "topology": copy.deepcopy(topology),
                "source_label": "上传数据文件",
                "source_kind": "upload",
                "project_name": quick_project_name or st.session_state.project_name,
                "village_name": quick_village_name or st.session_state.village_name,
                "data_version": quick_data_version or st.session_state.data_version,
            })
        except Exception as exc:
            st.error(f"上传数据载入失败：{exc}")
    return quick_analysis_requirements("upload")


def render_quick_analysis() -> None:
    st.markdown("### 快速分析")
    st.caption("采用“输入—计算—输出”单页结构，不显示四阶段导航；与引导式研判共用同一套数据模型、约束处理、统一评价链、求解器和导出结构。")
    quick_keys = initialize_quick_form_state()
    left, right = st.columns([1.2, 1.8])
    with left:
        st.markdown("<div class='work-card'><b>输入区域</b></div>", unsafe_allow_html=True)
        st.text_input("项目名称", key=quick_keys["project_name"])
        st.text_input("研究区名称", key=quick_keys["village_name"])
        st.text_input("数据版本", key=quick_keys["data_version"])
        ready, blockers = quick_source_selector()
        with st.expander("核心约束", expanded=True):
            solver_controls(compact=True, key_map=quick_keys)
            st.caption(f"参数来源：{st.session_state.scenario_parameter_source}")
        with st.expander("高级设置", expanded=False):
            st.caption("高级参数仅在需要时调整；未改变统一评价函数、约束定义和 Pareto 筛选逻辑。")
            st.slider("排水组合协同系数", 1.0, 1.3, step=.01, key="drainage_synergy")
            st.slider("第二项措施效果折减", .5, 1.0, step=.01, key="second_factor")
            st.selectbox("每单元最多措施数", [1, 2], key=quick_keys["max_measures"])
            st.checkbox("允许中高风险单元不治理", key=quick_keys["allow_untreated"])
        changed_fields = sync_quick_form_state(quick_keys)
        if changed_fields:
            invalidate_results("输入参数已变化，请重新计算。")
        quick_run = st.button("开始快速分析", type="primary", use_container_width=True, disabled=not ready)
        if blockers:
            st.markdown("<div class='invalid'><b>当前不能开始快速分析：</b><br>" + "<br>".join(blockers[:6]) + "</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='work-card'><b>结果区域</b></div>", unsafe_allow_html=True)
        summary = current_data_summary()
        active_source = st.session_state.data_source_kind == selected_quick_source_kind()
        if active_source:
            s1, s2, s3 = st.columns(3)
            s1.metric("治理单元数量", summary["unit_count"])
            s2.metric("风险路径数量", summary["path_count"])
            s3.metric("字段完整率", f"{summary['field_completeness']:.1f}%")
            st.markdown(f"<div class='work-card'><b>数据摘要</b><br>来源：{summary['data_source']}<br>空间范围：{summary['spatial_bounds']}<br>措施库规模：{summary['measure_count']} 项<br>参数依据状态：{summary['parameter_status']}</div>", unsafe_allow_html=True)
            render_map([], "quick_summary_map", show_units=False)
        else:
            st.markdown("<div class='notice'><b>等待载入数据</b><br>当前所选数据源尚未实际载入，地图预览、可行方案和 Pareto 结果暂不显示。</div>", unsafe_allow_html=True)
        if blockers:
            st.markdown("<div class='invalid'>当前数据或项目信息尚未满足快速分析要求，请先补齐输入后再运行。</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ok'>当前数据通过基础校验，可直接进入统一评价与多目标优化。</div>", unsafe_allow_html=True)
        if quick_run:
            run_model()
        result = st.session_state.result if st.session_state.result_valid and ready else None
        if result is None:
            if st.session_state.result_invalid_reason:
                st.markdown(f"<div class='notice'><b>结果已失效</b><br>{st.session_state.result_invalid_reason}</div>", unsafe_allow_html=True)
            st.markdown("<div class='notice'><b>空状态</b><br>尚未完成优化计算。请确认数据来源、预算、安全底线、求解器与数据校验状态后运行。</div>", unsafe_allow_html=True)
        else:
            plan_id, _ = selected_plan(result)
            summary = result.representatives[result.representatives.plan_id.astype(str) == plan_id].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("可行方案", f"{result.run_metadata['feasible_plans']:,}")
            c2.metric("Pareto候选", f"{result.run_metadata['pareto_plans']:,}")
            c3.metric("当前代表方案", str(summary["recommendation_type"]))
            render_plans(result)


def guide_footer() -> None:
    if not is_guided_mode():
        return
    step = int(st.session_state.current_step)
    valid, errors = step_validation(step, project_name=st.session_state.project_name, village_name=st.session_state.village_name,
                                    units_df=st.session_state.units_df, measures_df=st.session_state.measures_df, result_valid=st.session_state.result_valid)
    st.divider(); left, save, right = st.columns([1, 1, 1])
    if step > 0 and left.button("← 上一步", use_container_width=True):
        go_to_guided_step(step - 1)
    if save.button("保存本步", use_container_width=True):
        if valid:
            st.session_state.completed_steps.add(step)
            if step == 1:
                save_guided_snapshot()
            st.success("本步校验通过并已保存。")
        else: st.error(" ".join(errors))
    if step < len(GUIDED_STEPS) - 1 and right.button("下一步 →", type="primary", use_container_width=True):
        target = next_step(step, valid)
        if target == step: st.error("不能进入下一步：" + " ".join(errors))
        else:
            st.session_state.completed_steps.add(step)
            if step == 1:
                save_guided_snapshot()
            go_to_guided_step(target)


def main() -> None:
    st.set_page_config(page_title="综合整治工程技术多目标优化智能研判与决策平台", page_icon="⛰", layout="wide", initial_sidebar_state="expanded")
    inject_style(); initialize_state(); apply_pending_scenario(); apply_pending_navigation(); refresh_result_state()
    quick, run_clicked, _ = sidebar()
    header(); navigation()
    render_step_banner()
    if quick or run_clicked: run_model()
    result = st.session_state.result if st.session_state.result_valid else None
    if is_home_mode():
        render_home_page()
        return
    if is_quick_mode():
        render_quick_analysis()
        return
    if is_validation_mode():
        render_algorithm_validation()
        return
    step = int(st.session_state.current_step)
    if step == 0:
        render_project_overview(result)
    elif step == 1:
        render_data_units(result)
    elif step == 2:
        render_screening()
    elif step == 3:
        render_multidimensional_evaluation()
    elif step == 4:
        render_solver(result)
    elif step == 5:
        render_plans(result)
    elif step == 6:
        render_coordination(result)
    else:
        render_results(result)
    guide_footer()
