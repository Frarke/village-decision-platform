"""正式空间决策入口：python -m streamlit run v1_app.py。"""
import copy
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from model.result_exporter import (composition_figure, export_zip,
                                   pareto_figure, radar_figure)
from model.service import input_hash, load_geographic_sample, optimize_request, prepare_request
from model.input_package import read_input_package
from model.geographic_map import geographic_map_html, result_map_html


@st.cache_data(show_spinner=False)
def prepare_cached(request):
    return prepare_request(request)


def main():
    st.set_page_config(page_title="村域综合整治工程技术智能研判与决策", page_icon="⛰", layout="wide")
    st.markdown("""<style>.block-container{max-width:1560px;padding-top:2rem}h1{font-size:1.8rem!important;color:#173f49}
    [data-testid=stMetric]{border:1px solid #dce8e4;border-radius:8px;padding:14px;background:#f7faf9}
    [data-testid=stMetricValue]{font-size:1.7rem}div.stButton>button[kind=primary]{background:#197278;border:0}
    </style>""", unsafe_allow_html=True)
    st.title("村域综合整治工程技术智能研判与决策")
    st.caption("空间评价成果 → 重点治理区域 → 治理单元 → 多目标优化 → 空间治理方案")
    with st.expander("我需要提供哪些资料？计算后能拿到什么？", expanded=False):
        st.markdown("**提供已有空间评价成果与工程条件，治理单元由本模块计算。** 最终返回治理范围、各单元措施配置、措施图标位置、五类方案及指标，供其他系统直接显示。")
        st.dataframe(pd.DataFrame([
            ("研究范围", "boundary.geojson", "确定在哪个真实地理范围内计算"),
            ("已有风险、危险性评价", "risk.tif / haz.tif", "识别重点区域并筛选措施；不是普通截图"),
            ("易发性、易损性成果", "susceptibility.tif / vulnerability.tif", "上游已有评价输入，用于空间解释和单元统计；不暗中换算风险"),
            ("坡度及承灾体距离", "slope_deg.tif / distance_m.tif", "地形与邻近关系判断"),
            ("工程调查条件", "profile_id.tif + engineering_profiles.csv", "稳定性、坡高、施工空间、排水及锚固条件"),
            ("显式空间权重和风险路径", "decision_context.geojson", "按坐标关联生成单元，无需预填U01"),
            ("预算及计算配置", "project.json", "约束及规则配置")
        ], columns=["资料", "文件", "用途"]), hide_index=True, use_container_width=True)
        st.caption("接入方式：POST /optimize-package 上传ZIP，或POST /optimize提交JSON。返回GeoJSON和指标；成果HTML可直接嵌入宿主页面。真实底图需要网络，空间运算不依赖底图瓦片。")
    with st.sidebar:
        st.subheader("项目输入")
        mode = st.radio("展示模式", ["正式展示", "开发调试"], key="v1_mode")
        source = st.selectbox("数据来源", ["真实地点模拟资料", "上传资料包（ZIP）", "上传项目数据"], key="v1_source")
        uploaded = st.file_uploader("资料文件（ZIP或JSON）", type=["zip", "json"], key="v1_upload") if source != "真实地点模拟资料" else None
        st.download_button("下载分文件资料模板包", (ROOT/"data/v1_1/peitian_input_package.zip").read_bytes(),
                           "peitian_input_package.zip", mime="application/zip")
        if source != "真实地点模拟资料" and uploaded is None:
            st.info("请上传资料包或完整项目JSON。资料包内分别保存范围、评价栅格、调查表、空间权重和路径。")
            st.stop()
        try:
            request = (read_input_package(uploaded.getvalue()) if uploaded.name.lower().endswith(".zip")
                       else json.loads(uploaded.getvalue())) if uploaded else load_geographic_sample()
            request = copy.deepcopy(request)
            initial_solver = request.get("configuration", {}).get("solver", {})
            # 不同输入使用独立控件状态，避免把上一个项目的预算带入新数据。
            source_key = input_hash(request)[:12]
            st.write(request.get("village", {}).get("name", "未命名研究区"))
            with st.expander("计算设置", expanded=False):
                budget = st.number_input("总预算（万元）", min_value=0.0, value=float(initial_solver.get("total_budget", 520)),
                                         key=f"v1_budget_{source_key}")
                method_default = request.get("configuration", {}).get("method", "nsga2")
                method = st.selectbox("求解方式", ["NSGA-II 多目标搜索", "精确枚举"],
                                      index=0 if method_default == "nsga2" else 1, key=f"v1_method_{source_key}")
            request.setdefault("configuration", {}).setdefault("solver", {})["total_budget"] = budget
            request["configuration"]["method"] = "nsga2" if method.startswith("NSGA") else "exact"
            current_hash = input_hash(request)
            prepared = prepare_cached(request)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"输入校验未通过：{exc}")
            st.stop()
        st.caption("NSGA-II：第二代非支配排序遗传算法，用于搜索兼顾四项目标的非劣方案。")
        run = st.button("运行综合研判", type="primary", use_container_width=True, key="v1_run")
        st.caption("空间参数及措施效应仍需结合工程资料标定。")
    data_status = request["village"]["data_status"]
    st.info("模拟示例数据 · 人工风险路径 · 计算结果用于方法与流程验证" if data_status == "MOCK"
            else "已载入研究区数据 · 参数依据随输入记录 · 方案须经过工程复核")
    if run:
        try:
            with st.spinner("正在识别重点区域、生成治理单元并实际计算候选方案……"):
                result = optimize_request(request)
            st.session_state.v1_result = result
            st.session_state.v1_result_hash = current_hash
            st.session_state.v1_selected_unit = prepared["required_risk_weight_ids"][0] if prepared["required_risk_weight_ids"] else None
            st.session_state.v1_selection_event = None
        except (ValueError, TypeError, KeyError) as exc:
            logging.getLogger(__name__).warning("研判校验未通过：%s", exc)
            st.session_state.v1_result_hash = None
            st.error(f"本次计算未完成：{exc}")
    result = st.session_state.get("v1_result")
    if result is not None and st.session_state.get("v1_result_hash") != current_hash:
        st.warning("输入或计算设置已变化，原结果已失效，请重新计算。")
        result = None
    if result is None:
        st.subheader("输入校验与治理对象识别")
        metrics = prepared["spatial_metrics"]
        columns = st.columns(3)
        columns[0].metric("有效评价片区", metrics["valid_features"])
        columns[1].metric("重点治理面积（公顷）", f"{metrics['priority_area_m2']/10000:.2f}")
        columns[2].metric("自动生成单元", metrics["governance_unit_count"])
        st.write("资料校验通过。下图可查看已有评价、重点区域及生成单元的空间关系；点击运行后产生工程措施配置。")
        st.caption(f"研究范围内有效像元{metrics['clipped_features']}个，重点像元{metrics['priority_features']}个；单元由选中像元连通形成。研究范围外{metrics['outside_features']}个像元已排除。")
        components.html(geographic_map_html(prepared, request, embedded=True), height=735, scrolling=True)
        if metrics["filtered_area_m2"]:
            st.caption(f"小斑块过滤面积：{metrics['filtered_area_m2']:.1f}平方米。")
        if mode == "开发调试":
            st.json(prepared)
        return
    if result["status"] != "completed":
        st.warning(result["summary"]["message"])
        st.json(result["spatial_metrics"])
        return
    labels = [s["label"] for s in result["representative_solutions"]]
    label = st.radio("代表方案", labels, horizontal=True, key="v1_plan_label")
    solution = next(s for s in result["representative_solutions"] if s["label"] == label)
    metrics = st.columns(5)
    metrics[0].metric("全寿命成本（万元）", f"{solution['total_cost']:.1f}")
    for col, (key, name) in zip(metrics[1:], (("safety", "安全性"), ("economy", "经济性"),
                                           ("sustainability", "可持续性"), ("resilience", "韧性"))):
        col.metric(f"{name}（分）", f"{solution['objectives'][key]:.1f}")
    st.subheader("真实地图上的治理措施配置")
    ids = [f["properties"]["unit_id"] for f in result["governance_units"]["features"]]
    if st.session_state.get("v1_selected_unit") not in ids:
        st.session_state.v1_selected_unit = ids[0]
    st.selectbox("定位治理单元（也可直接点击地图和措施图标）", ids, key="v1_selected_unit")
    components.html(result_map_html(result, label, st.session_state.v1_selected_unit, embedded=True), height=735, scrolling=True)
    st.write("推荐依据：" + solution["basis"])
    st.caption(f"关键路径完整覆盖率 {solution['path_coverage']:.1f}%；区域协同已进入每个候选方案的四目标评价。")
    st.subheader("方案比较")
    st.caption("Pareto（帕累托）非劣方案：无法在不降低其他目标的同时提高某一目标。四目标均为越大越好；二维投影不代表完整支配关系。")
    charts = st.columns(2)
    with charts[0]:
        st.plotly_chart(radar_figure(result), use_container_width=True, key="v1_radar")
    with charts[1]:
        st.plotly_chart(pareto_figure(result, label), use_container_width=True, key="v1_pareto")
    with st.expander("工程措施组成与完整非劣方案"):
        st.plotly_chart(composition_figure(result, label), use_container_width=True, key="v1_composition")
        st.dataframe(pd.DataFrame(result["pareto_front"])[["plan_id", "safety", "economy", "sustainability", "resilience", "total_cost"]]
                     .rename(columns={"plan_id": "方案编号", "safety": "安全性", "economy": "经济性", "sustainability": "可持续性",
                                      "resilience": "韧性", "total_cost": "全寿命成本（万元）"}), hide_index=True, use_container_width=True)
    st.subheader("成果输出")
    cols = st.columns(3)
    cols[0].download_button("下载完整成果包", export_zip(result), "village_decision_results.zip", mime="application/zip")
    cols[1].download_button("下载真实底图与工程图标地图", result_map_html(result), "spatial_decision_map.html", mime="text/html")
    cols[2].download_button("下载当前方案空间数据", json.dumps(result["map_result"]["solutions"][label], ensure_ascii=False, indent=2),
                            f"{label}.geojson", mime="application/geo+json")
    st.caption("；".join(result["summary"]["notices"]))
    if mode == "开发调试":
        st.subheader("开发调试记录")
        optimization = result["metrics"]["optimization"]
        st.write(f"可行方案：{optimization['feasible_plans']}；非劣方案：{optimization['pareto_plans']}；实际用时：{optimization['runtime_seconds']:.3f}秒")
        tabs = st.tabs(["单元与措施", "真实迭代", "输入与依据"])
        with tabs[0]:
            st.dataframe(pd.DataFrame([f["properties"] for f in result["governance_units"]["features"]]), hide_index=True)
            st.dataframe(pd.DataFrame(result["screening"]), hide_index=True)
        with tabs[1]:
            st.dataframe(pd.DataFrame(result["metrics"]["iteration_history"]), hide_index=True)
        with tabs[2]:
            st.json(result["provenance"])


if __name__ == "__main__":
    main()
