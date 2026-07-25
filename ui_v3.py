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

from model.benchmarks import validate_standard_functions
from model.data_validation import validate_project_data
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
MODULES = ("项目总览", "数据与单元", "智能研判", "方案比选", "空间协同", "成果管理")
STEP_MODULE = ("项目总览", "数据与单元", "智能研判", "智能研判", "智能研判", "方案比选", "空间协同", "成果管理")
COLORS = {"navy": "#102A43", "blue": "#173F5F", "cyan": "#168AAD", "muted": "#6B7C8F", "paper": "#F4F7FA", "red": "#C0392B", "orange": "#D97706", "green": "#2E7D5B"}
RISK_COLORS = {1: "#DDEFE5", 2: "#A9D8C0", 3: "#F2C66D", 4: "#E78856", 5: "#C0392B"}


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
def load_defaults() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    return (
        pd.read_csv(DATA_DIR / "default_units.csv"),
        pd.read_csv(DATA_DIR / "default_measures.csv"),
        load_geojson(DATA_DIR / "spatial_units.geojson"),
        load_geojson(DATA_DIR / "spatial_context.geojson"),
        load_topology(DATA_DIR / "spatial_topology.json"),
    )


def state_defaults() -> dict[str, object]:
    units, measures, units_geojson, context_geojson, topology = load_defaults()
    return {
        "mode": "引导式研判", "current_step": 0, "current_module": "项目总览", "module_nav": "项目总览", "pending_module": None, "completed_steps": set(),
        "intelligence_page": "工程措施准入", "last_intelligence_step": -1,
        "project_name": "村庄地质灾害综合整治研究", "village_name": "示例研究区", "data_version": "V3-示例数据-20260722",
        "units_df": units.copy(), "measures_df": measures.copy(), "units_geojson": copy.deepcopy(units_geojson),
        "context_geojson": copy.deepcopy(context_geojson), "topology": copy.deepcopy(topology), "editor_version": 0,
        "selected_unit": "U01", "unit_selector_widget": "U01", "pending_unit_selector": None, "selected_plan_label": "综合折中方案",
        "map_background": "在线街道底图", "display_mode": "治理前风险", "show_paths": True,
        "result": None, "exact_result": None, "nsga_result": None, "validation": None, "benchmark_results": None,
        "result_valid": False, "last_optimization_hash": None, "last_preference_hash": None, "scenario_history": [],
        "total_budget": 520.0, "max_measures": 2, "allow_untreated": False, "min_fs": 1.05, "min_safety": 58.0,
        "max_enumerations": 200_000, "drainage_synergy": 1.10, "second_factor": 0.90,
        "weight_safety": 0.35, "weight_economy": 0.20, "weight_sustainability": 0.20, "weight_resilience": 0.25,
        "solver_method": "exact", "population_size": 60, "generations": 25, "crossover_probability": .90,
        "mutation_probability": .20, "random_seed": 42, "verify_current_run": False,
    }


def initialize_state() -> None:
    for key, value in state_defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    return optimization_hash(units, measures, make_config(), st.session_state.solver_method)


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
            st.session_state.result_valid = False
            return
        new_pref = current_preference_hash()
        if st.session_state.last_preference_hash != new_pref:
            for result in (st.session_state.exact_result, st.session_state.nsga_result):
                rebuild_representatives(result)
            st.session_state.result = st.session_state.exact_result if st.session_state.solver_method == "exact" else st.session_state.nsga_result
            st.session_state.last_preference_hash = new_pref
    except (KeyError, TypeError, ValueError):
        st.session_state.result_valid = False


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


def selected_plan(result) -> tuple[str, list[dict[str, object]]]:
    if result is None or result.representatives.empty:
        return "", []
    labels = result.representatives["recommendation_type"].astype(str).tolist()
    if st.session_state.selected_plan_label not in labels:
        st.session_state.selected_plan_label = "综合折中方案" if "综合折中方案" in labels else labels[0]
    row = result.representatives[result.representatives["recommendation_type"] == st.session_state.selected_plan_label].iloc[0]
    plan_id = str(row["plan_id"])
    return plan_id, result.detail_lookup.get(plan_id, [])


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


def map_figure(plan_details: list[dict[str, object]] | None, selected_unit: str, online: bool) -> go.Figure:
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
    center = {"lon": float(frame.longitude.mean()), "lat": float(frame.latitude.mean())}
    common = dict(geojson=st.session_state.units_geojson, locations=frame.unit_id, z=frame.display,
                  featureidkey="properties.unit_id", colorscale=scale, marker_line_width=1.6,
                  customdata=np.stack([frame.unit_id, frame.unit_name], axis=-1),
                  hovertemplate="治理单元：%{customdata[0]}<br>%{customdata[1]}<br>" + label + "：%{z:.1f}<extra></extra>")
    if online:
        fig = go.Figure(go.Choroplethmap(**common, marker_opacity=.67, colorbar_title=label))
        scatter = go.Scattermap
    else:
        fig = go.Figure(go.Choropleth(**common, colorbar_title=label))
        scatter = go.Scattergeo
    # 中心点是稳定的点击命中区；面要素与中心点都携带唯一unit_id。
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
        fig.update_layout(map={"style": "open-street-map", "center": center, "zoom": 13.2})
    else:
        fig.update_geos(fitbounds="locations", visible=True, showland=True, landcolor="#E8EEF2", showcoastlines=False,
                        lonaxis={"showgrid": True, "gridcolor": "#C9D5DE"}, lataxis={"showgrid": True, "gridcolor": "#C9D5DE"})
    fig.update_layout(height=585, margin={"l": 0, "r": 0, "t": 32, "b": 0}, showlegend=False,
                      title={"text": f"村庄级空间研判｜{mode}", "x": .01, "font": {"size": 15, "color": COLORS["navy"]}},
                      clickmode="event+select", uirevision="v3-map-view", selectionrevision="v3-map-selection")
    return fig


def render_map(plan_details: list[dict[str, object]] | None, key: str) -> None:
    ids = st.session_state.units_df.unit_id.astype(str).tolist()
    st.session_state.selected_unit = synchronize_selected_unit(st.session_state.selected_unit, ids)
    online = st.session_state.map_background == "在线街道底图"
    try:
        event = st.plotly_chart(map_figure(plan_details, st.session_state.selected_unit, online), use_container_width=True,
                                key=key, on_select="rerun", selection_mode="points", config={"displaylogo": False, "scrollZoom": True})
        clicked = extract_unit_id(event, ids)
        if clicked and clicked != st.session_state.selected_unit:
            st.session_state.selected_unit = clicked
            st.session_state.pending_unit_selector = clicked
            st.rerun()
    except Exception as exc:
        if online:
            st.warning(f"在线街道底图不可用，已自动切换到离线坐标网格。治理单元和属性仍可使用。原因：{exc}")
            try:
                event = st.plotly_chart(map_figure(plan_details, st.session_state.selected_unit, False), use_container_width=True,
                                        key=f"{key}_offline", on_select="rerun", selection_mode="points", config={"displaylogo": False, "scrollZoom": True})
                clicked = extract_unit_id(event, ids)
                if clicked and clicked != st.session_state.selected_unit:
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
    st.caption("当前空间单元和风险路径为平台演示数据，不代表现场实测边界。")
    st.markdown(f"**单元类型：**{props.get('hazard_type', '待补充')}  \n**空间位置：**东经{props.get('longitude', 0):.5f}°，北纬{props.get('latitude', 0):.5f}°  \n**岩土体类型：**{props.get('material_type', '待补充')}  \n**坡高 / 坡度 / 坡向：**{row['height_m']:g}米 / {row['slope_deg']:g}° / 待补充  \n**现状稳定系数：**{row['fs0']:.2f}；**危险性 / 风险：**{int(row['haz'])}级 / {int(row['risk'])}级  \n**主要承灾体：**{props.get('protected_object', '待补充')}")
    st.markdown(f"**自动识别标签：**{tagged['auto_tags']}  \n**标签判定依据：**{tagged['tag_evidence']}")
    screening = None
    if result is not None:
        screening = result.screening[result.screening.unit_id.astype(str) == selected]
    else:
        units, measures = model_inputs(); frame = pd.DataFrame([item.to_dict() for item in screen_all(units, measures)])
        screening = frame[frame.unit_id.astype(str) == selected]
    admitted = "、".join(screening.loc[screening.eligible, "measure_name"].astype(str)) or "无"
    excluded = "；\n\n".join(f"{item.measure_name}：{item.exclusion_reason}" for item in screening.loc[~screening.eligible].itertuples()) or "无"
    st.markdown(f"**准入措施：**{admitted}  \n**被剔除措施及原因：**  \n{excluded}")
    if detail:
        unit = GovernanceUnit.from_record(row.to_dict()); before = evaluate_unit(unit, (), make_config())
        st.markdown(f"**当前方案配置：**{detail['measures']}  \n**治理前 / 后稳定系数：**{before['fs_before_rain']:.2f} / {detail['fs_rain']:.2f}  \n**治理前 / 后安全得分：**{before['safety']:.1f} / {detail['safety']:.1f}  \n**治理后残余危险性 / 风险：**{detail['residual_hazard']}级 / {detail['residual_risk']}级  \n**风险降低幅度：**{detail['risk_reduction']:.2f}级  \n**估算全寿命成本：**{detail['lifecycle_cost']:.1f}万元  \n**未消除或转移风险：**{detail['downstream_transfer_review']}  \n**专家复核事项：**{detail['expert_review']}  \n**数据来源与依据：**{row['evidence_level']}；{row['notes']}")


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
            exact = solve_exact(units, measures, config, DATA_DIR / "assumptions.csv")
        if st.session_state.solver_method == "nsga2" or verify:
            bar = st.progress(0.0, text="正在初始化离散方案种群……")
            def callback(item: dict[str, object]) -> None:
                ratio = float(item["generation"]) / max(config.nsga_generations, 1)
                bar.progress(ratio, text=f"真实迭代 {item['generation']}/{config.nsga_generations}｜可行个体{item['current_feasible']}｜累计评价{item['unique_evaluations']}个唯一方案")
            nsga = solve_nsga2(units, measures, config, DATA_DIR / "assumptions.csv", progress_callback=callback)
            bar.empty()
        result = exact if st.session_state.solver_method == "exact" else nsga
        st.session_state.exact_result = exact; st.session_state.nsga_result = nsga; st.session_state.result = result
        st.session_state.validation = validate_runs(exact, [nsga]) if exact is not None and nsga is not None else None
        st.session_state.result_valid = True; st.session_state.last_optimization_hash = current_optimization_hash(); st.session_state.last_preference_hash = current_preference_hash()
        st.session_state.completed_steps.add(4); st.session_state.scenario_history.append(result.run_metadata)
        holder.success(f"计算完成：{result.run_metadata['feasible_plans']:,}个可行方案，{result.run_metadata['pareto_plans']:,}个Pareto方案。")
    except Exception as exc:
        st.session_state.result_valid = False; holder.error(f"计算未完成：{exc}")


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
    holder = st.empty(); holder.info("正在运行ZDT1、ZDT2、ZDT3和DTLZ2标准函数的3随机种子验证……")
    frame, examples = validate_standard_functions()
    st.session_state.benchmark_results = {"summary": frame, "examples": examples}
    holder.success("4个标准函数、12次独立运行已完成。")


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
        st.markdown("<div class='platform-head'><h1>村庄地质灾害综合整治GIS智能研判与多目标决策平台</h1><div class='platform-sub'>数据核查 · 工程准入 · 多目标优化 · 空间协同 · 专家复核</div></div>", unsafe_allow_html=True)
    with right:
        st.markdown(f"<div class='platform-head meta-line'>项目：{st.session_state.project_name}<br>村庄：{st.session_state.village_name}｜场景：{scenario}<br>数据：{st.session_state.data_version}｜{status}</div>", unsafe_allow_html=True)
    st.markdown("<div class='notice'><b>数据边界：</b>当前空间单元和风险路径为平台演示数据，不代表现场实测边界；估算成本用于方案比选，不等同于施工预算。</div>", unsafe_allow_html=True)
    if st.session_state.result is not None and not st.session_state.result_valid:
        st.markdown("<div class='invalid'>当前输入条件已发生变化，现有结果已经失效，请重新运行优化计算。</div>", unsafe_allow_html=True)


def sidebar() -> tuple[bool, bool, bool]:
    quick = run_clicked = validate_clicked = False
    with st.sidebar:
        st.markdown("## 当前任务")
        st.radio("工作模式", ["引导式研判", "快速分析"], key="mode", horizontal=True)
        if st.session_state.mode == "引导式研判":
            step = int(st.session_state.current_step)
            progress_lines = []
            for i, label in enumerate(GUIDED_STEPS):
                cls = "step-now" if i == step else ("step-done" if i in st.session_state.completed_steps else "")
                marker = "●" if i == step else ("✓" if i in st.session_state.completed_steps else "○")
                progress_lines.append(f"<div class='{cls}'>{marker} {i+1}. {label}</div>")
            st.markdown("<div class='step-list'>" + "".join(progress_lines) + "</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='task-card'><b>本步目标</b><br>{STEP_OUTPUTS[step]}</div>", unsafe_allow_html=True)
            if step == 0:
                st.text_input("当前项目名称*", key="project_name"); st.text_input("村庄或研究区*", key="village_name"); st.text_input("数据版本", key="data_version")
            elif step == 1:
                st.caption("核对治理单元属性，地图点击后可在主工作区查看完整详情。")
            elif step == 2:
                st.caption("逐条核查4个单元×8类措施形成的32条准入判定记录。")
            elif step in {3, 4}:
                solver_controls()
                if step == 4: run_clicked = st.button("运行优化计算", type="primary", use_container_width=True)
            elif step == 5:
                st.caption("在主工作区调整代表方案偏好；权重变化不会重新运行优化。")
            elif step == 6:
                map_controls()
            elif step == 7:
                st.caption("复核推荐方案、参数依据、数据边界与专家复核事项后导出。")
        else:
            st.markdown("<div class='task-card'><b>快速分析</b><br>适用于已确认数据与参数的场景；不会默认运行耗时的算法验证。</div>", unsafe_allow_html=True)
            st.text_input("当前项目", key="project_name"); st.text_input("当前村庄", key="village_name")
            solver_controls(compact=True); quick = st.button("一键运行当前场景", type="primary", use_container_width=True)
        st.divider(); st.button("恢复默认示例场景", use_container_width=True, on_click=reset_defaults)
    return quick, run_clicked, validate_clicked


def solver_controls(compact: bool = False) -> None:
    st.selectbox("求解方法", ["exact", "nsga2"], format_func=lambda x: "精确枚举" if x == "exact" else "NSGA-II智能搜索", key="solver_method")
    st.number_input("总预算（万元）", 0.0, 100000.0, step=10.0, key="total_budget", help="约束全村各单元的估算全寿命周期成本总和。")
    st.number_input("治理后稳定系数下限", .5, 2.0, step=.01, key="min_fs")
    st.number_input("最低安全得分", 0.0, 100.0, step=1.0, key="min_safety", help="综合稳定性和灾害链控制能力形成的0—100分指标；默认58分为示例阈值，仍需项目标定。")
    if not compact:
        with st.expander("组合与作用参数"):
            st.selectbox("每单元最多措施数", [1, 2], key="max_measures"); st.checkbox("允许中高风险单元不治理", key="allow_untreated")
            st.slider("排水组合协同系数", 1.0, 1.3, step=.01, key="drainage_synergy"); st.slider("第二项措施效果折减", .5, 1.0, step=.01, key="second_factor")
    if st.session_state.solver_method == "nsga2":
        with st.expander("NSGA-II搜索参数", expanded=True):
            st.number_input("种群规模", 20, 1000, step=10, key="population_size"); st.number_input("最大迭代代数", 1, 1000, step=5, key="generations")
            st.slider("交叉概率", 0.0, 1.0, step=.05, key="crossover_probability"); st.slider("变异概率", 0.0, 1.0, step=.05, key="mutation_probability")
            st.number_input("随机种子", 0, 999999, step=1, key="random_seed")
    st.checkbox("同时运行精确基准进行本次对照", key="verify_current_run", help="仅在需要算法验证时启用；快速分析默认关闭。")


def preference_controls() -> None:
    st.caption("偏好权重只改变Pareto集内的代表方案选择，不重新运行优化。")
    st.slider("安全偏好", 0.0, 1.0, step=.05, key="weight_safety"); st.slider("经济偏好", 0.0, 1.0, step=.05, key="weight_economy")
    st.slider("可持续偏好", 0.0, 1.0, step=.05, key="weight_sustainability"); st.slider("韧性偏好", 0.0, 1.0, step=.05, key="weight_resilience")


def map_controls() -> None:
    st.selectbox("底图", ["在线街道底图", "离线坐标网格"], key="map_background")
    st.selectbox("地图表达", ["治理前风险", "治理后安全性", "推荐措施", "约束未满足单元"], key="display_mode")
    st.checkbox("显示关键风险路径", key="show_paths")


def navigation() -> None:
    if st.session_state.pending_module:
        st.session_state.module_nav = st.session_state.pending_module
        st.session_state.current_module = st.session_state.pending_module
        st.session_state.pending_module = None
    module = st.radio("一级模块", MODULES, horizontal=True, key="module_nav", label_visibility="collapsed")
    if module != st.session_state.current_module:
        st.session_state.current_module = module
    if st.session_state.mode == "引导式研判":
        matching = [i for i, item in enumerate(STEP_MODULE) if item == module]
        if matching and st.session_state.current_step not in matching:
            st.session_state.current_step = matching[0]


def render_project_overview(result) -> None:
    _, details = selected_plan(result)
    left, right = st.columns([3, 1.15])
    with left:
        render_map(details, "overview_map")
        st.markdown("<div class='legend-row'><span><i class='legend-dot' style='background:#C0392B'></i>高风险/未控制</span><span><i class='legend-dot' style='background:#D97706'></i>部分控制/待复核</span><span><i class='legend-dot' style='background:#2E7D5B'></i>满足控制</span><span>箭头语义：物源 → 输移 → 承灾体</span></div>", unsafe_allow_html=True)
    with right:
        st.markdown("### 研究区概况")
        a, b = st.columns(2); a.metric("治理单元", len(st.session_state.units_df)); b.metric("候选措施", len(st.session_state.measures_df))
        c, d = st.columns(2); c.metric("关键路径", len(st.session_state.topology.get("paths", []))); d.metric("结果状态", "有效" if result else "待计算")
        st.markdown("<div class='work-card'><b>研判目标</b><br>在工程准入、安全底线和预算约束下，比较安全性、经济性、可持续性与韧性，形成可解释、可复核的村域治理方案。</div>", unsafe_allow_html=True)
        unit_detail(result, details)


def render_data_units(result) -> None:
    _, details = selected_plan(result)
    map_controls_inline = st.columns([1, 1, 1, 3])
    with map_controls_inline[0]: st.selectbox("底图类型", ["在线街道底图", "离线坐标网格"], key="map_background")
    with map_controls_inline[1]: st.selectbox("表达模式", ["治理前风险", "治理后安全性", "推荐措施", "约束未满足单元"], key="display_mode")
    with map_controls_inline[2]: st.checkbox("风险路径", key="show_paths")
    left, right = st.columns([2.8, 1.25])
    with left: render_map(details, "units_map")
    with right: unit_detail(result, details)
    linked_unit_table()
    with st.expander("编辑或导入项目数据"):
        st.info("导入后会校验必填字段、类型、编号唯一性、GeoJSON对应关系和风险路径引用。")
        unit_file = st.file_uploader("治理单元CSV", type=["csv"], key="unit_upload")
        measure_file = st.file_uploader("措施库CSV", type=["csv"], key="measure_upload")
        geo_file = st.file_uploader("治理单元GeoJSON", type=["geojson", "json"], key="geo_upload")
        topo_file = st.file_uploader("风险路径JSON", type=["json"], key="topo_upload")
        if st.button("校验并载入所选数据"):
            try:
                units_df = pd.read_csv(unit_file) if unit_file else st.session_state.units_df
                measures_df = pd.read_csv(measure_file) if measure_file else st.session_state.measures_df
                geojson = json.load(geo_file) if geo_file else st.session_state.units_geojson
                topology = json.load(topo_file) if topo_file else st.session_state.topology
                errors = validate_project_data(units_df, measures_df, geojson, topology)
                if errors: st.error("；".join(errors))
                else:
                    st.session_state.units_df = units_df; st.session_state.measures_df = measures_df; st.session_state.units_geojson = geojson; st.session_state.topology = topology
                    st.session_state.selected_unit = synchronize_selected_unit(st.session_state.selected_unit, units_df.unit_id.astype(str).tolist())
                    st.session_state.result_valid = False; st.success("数据通过一致性校验并已载入。")
            except Exception as exc: st.error(f"导入失败：{exc}")
        st.session_state.units_df = st.data_editor(st.session_state.units_df, key=f"units_editor_{st.session_state.editor_version}", use_container_width=True, num_rows="dynamic", column_config=editor_columns(st.session_state.units_df))


def render_screening() -> None:
    units, measures = model_inputs(); screening = pd.DataFrame([item.to_dict() for item in screen_all(units, measures)])
    st.markdown("### 工程措施准入判定")
    st.info(f"{len(screening)}条判定记录＝{len(units)}个治理单元×{len(measures)}类候选措施，并不表示{len(screening)}种不同工程措施。M01—M08是措施类别编码；每条判定使用Uxx-Mxx唯一编号。")
    a, b, c = st.columns(3); a.metric("判定记录", len(screening)); b.metric("准入", int(screening.eligible.sum())); c.metric("剔除", int((~screening.eligible).sum()))
    st.dataframe(chinese_frame(screening), use_container_width=True, hide_index=True)
    with st.expander("准入条件与符号说明"):
        st.markdown("- 汇流面积 *Ac*，单位为平方千米。例如“Ac<0.05 km²”表示汇流面积小于0.05平方千米的沟谷或小型汇水区。\n- 坡度 *θ*，单位为度；坡高 *H*、坡脚空间 *W*，单位为米。\n- 风险与危险性均采用1—5级，数值越高表示关注程度越高。每条记录同时给出命中条件、限制条件、准入结论和剔除原因。")


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
        if st.button("立即运行优化计算", type="primary", use_container_width=True): run_model()
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
        st.markdown("### 第一层：标准测试函数验证")
        st.info("标准函数验证算法的选择、交叉、变异和非支配搜索机制；不能代替工程离散编码与约束耦合验证。")
        if st.button("运行4个标准函数×3个随机种子", type="primary"): run_standard_validation()
        bundle = st.session_state.benchmark_results
        if bundle:
            st.dataframe(bundle["summary"], use_container_width=True, hide_index=True)
            name = st.selectbox("查看函数", ["ZDT1", "ZDT2", "ZDT3", "DTLZ2"])
            example = bundle["examples"][name]
            if example.objectives.shape[1] == 2:
                fig = go.Figure(); fig.add_trace(go.Scatter(x=example.theoretical_front[:, 0], y=example.theoretical_front[:, 1], mode="lines", name="理论Pareto前沿"))
                fig.add_trace(go.Scatter(x=example.objectives[:, 0], y=example.objectives[:, 1], mode="markers", name="算法解集")); fig.update_layout(title=f"{name}理论前沿与算法解集", xaxis_title="目标1", yaxis_title="目标2")
                st.plotly_chart(fig, use_container_width=True)
            fig = px.line(example.history, x="迭代代数", y="IGD", title=f"{name}收敛过程（IGD）"); st.plotly_chart(fig, use_container_width=True)
            st.caption("图表解读：IGD越低，说明算法解集与理论Pareto前沿的距离越小；Spacing反映分布均匀性，Hypervolume反映收敛与覆盖的综合表现。")
    with tabs[1]:
        st.markdown("### 第二层：村庄工程离散案例验证")
        st.info("精确枚举作为小案例完整基准，用于检验离散编码、准入/互斥、硬约束、修复和工程评价耦合是否正确。")
        if st.button("运行精确基准＋3随机种子NSGA-II", type="primary"): run_multiseed_validation()
        if st.session_state.validation:
            st.dataframe(st.session_state.validation.metrics, use_container_width=True, hide_index=True)
            st.dataframe(st.session_state.validation.seed_summary, use_container_width=True, hide_index=True)
            st.markdown(f"<div class='ok'>{st.session_state.validation.conclusion}</div>", unsafe_allow_html=True)


def render_plans(result) -> None:
    if not result:
        st.warning("请先完成优化计算，再进入方案比选。")
        return
    st.markdown("### Pareto候选方案与代表方案")
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
    cheapest = result.representatives.sort_values("total_cost").iloc[0]; metrics = coordination_metrics(st.session_state.topology, details)
    st.markdown(f"<div class='work-card'><b>推荐结论：</b>{st.session_state.selected_plan_label}（{plan_id}）。估算全寿命成本{selected['total_cost']:.1f}万元；相较代表方案中的最低成本方案增加{selected['total_cost']-cheapest['total_cost']:.1f}万元，安全性提高{selected['safety']-cheapest['safety']:.1f}分。<br><b>空间复核：</b>{metrics['未治理薄弱环节']}<br><b>参数复核：</b>{selected['expert_review']}<br><b>限制：</b>示例空间边界和待标定参数不能直接替代现场勘察与施工设计。</div>", unsafe_allow_html=True)
    st.dataframe(chinese_frame(pd.DataFrame(details)), use_container_width=True, hide_index=True)


def render_coordination(result) -> None:
    if not result:
        st.warning("请先完成优化计算，再开展空间协同研判。")
        return
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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("图表解读：只有措施类型与物源、输移或承灾体环节相匹配时才计为有效覆盖。当前协同指标用于结果解释和代表方案复核，尚未作为NSGA-II独立优化目标。")


def render_results(result) -> None:
    if not result:
        st.warning("暂无有效成果，请先运行优化计算。")
        return
    meta = result.run_metadata
    st.markdown("### 场景追踪与成果导出")
    st.dataframe(pd.DataFrame([{ "项目名称": st.session_state.project_name, "村庄名称": st.session_state.village_name, "数据版本": st.session_state.data_version,
        "场景编号": meta["scenario_id"], "创建时间": meta["run_time"], "配置标识": meta["configuration_hash"],
        "求解方法": "精确枚举" if meta["solver"] == "exact_enumeration" else "NSGA-II智能搜索", "随机种子": meta["random_seed"],
        "结果状态": "有效", "数据性质": "平台演示数据"}]), use_container_width=True, hide_index=True)
    st.download_button("下载当前结果JSON", result_to_json(result), file_name="当前场景完整结果.json", mime="application/json", use_container_width=True)
    st.download_button("下载科研分析报告HTML", build_html_report(result), file_name="科研分析报告.html", mime="text/html", use_container_width=True)
    st.download_button("下载完整成果包ZIP", export_zip(result), file_name="村庄综合整治研判成果.zip", mime="application/zip", use_container_width=True)
    st.markdown("<div class='notice'><b>专家复核清单：</b>措施效应与成本标定；排水出口及下游风险；抗滑桩滑面与嵌固段；示例GeoJSON替换；稳定系数和安全得分阈值复核。</div>", unsafe_allow_html=True)


def guide_footer() -> None:
    if st.session_state.mode != "引导式研判":
        return
    step = int(st.session_state.current_step)
    valid, errors = step_validation(step, project_name=st.session_state.project_name, village_name=st.session_state.village_name,
                                    units_df=st.session_state.units_df, measures_df=st.session_state.measures_df, result_valid=st.session_state.result_valid)
    st.divider(); left, save, right = st.columns([1, 1, 1])
    if step > 0 and left.button("← 上一步", use_container_width=True):
        st.session_state.current_step -= 1; st.session_state.pending_module = STEP_MODULE[st.session_state.current_step]; st.rerun()
    if save.button("保存本步", use_container_width=True):
        if valid: st.session_state.completed_steps.add(step); st.success("本步校验通过并已保存。")
        else: st.error(" ".join(errors))
    if step < len(GUIDED_STEPS) - 1 and right.button("下一步 →", type="primary", use_container_width=True):
        target = next_step(step, valid)
        if target == step: st.error("不能进入下一步：" + " ".join(errors))
        else:
            st.session_state.completed_steps.add(step); st.session_state.current_step = target; st.session_state.pending_module = STEP_MODULE[target]; st.rerun()


def main() -> None:
    st.set_page_config(page_title="村庄地质灾害综合整治GIS智能研判平台", page_icon="⛰", layout="wide", initial_sidebar_state="expanded")
    inject_style(); initialize_state(); refresh_result_state()
    quick, run_clicked, _ = sidebar()
    header(); navigation()
    if quick or run_clicked: run_model()
    result = st.session_state.result if st.session_state.result_valid else None
    module = st.session_state.current_module
    if module == "项目总览": render_project_overview(result)
    elif module == "数据与单元": render_data_units(result)
    elif module == "智能研判":
        if st.session_state.mode == "引导式研判" and st.session_state.last_intelligence_step != st.session_state.current_step:
            st.session_state.intelligence_page = "工程措施准入" if st.session_state.current_step == 2 else "优化计算"
            st.session_state.last_intelligence_step = st.session_state.current_step
        sub = st.radio("智能研判二级页面", ["工程措施准入", "优化计算", "算法验证"], horizontal=True, key="intelligence_page")
        if sub == "工程措施准入": render_screening()
        elif sub == "优化计算": render_solver(result)
        else: render_algorithm_validation()
    elif module == "方案比选": render_plans(result)
    elif module == "空间协同": render_coordination(result)
    else: render_results(result)
    guide_footer()
