"""同一结果快照驱动页面与离线成果；地图导出不依赖在线底图。"""
from __future__ import annotations

import html
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import plotly.graph_objects as go
from pyproj import CRS
from shapely.geometry import shape

from .data_adapter import reproject


COLORS = ["#197278", "#d17a34", "#576ca8", "#a64b65", "#6e9143", "#8c65a3", "#497d9c", "#ae8742"]
OBJECTIVE_NAMES = {"safety": "安全性", "economy": "经济性", "sustainability": "可持续性", "resilience": "韧性"}


def measure_colors(result):
    names = sorted({f["properties"]["measures"] for fc in result["map_result"]["solutions"].values() for f in fc["features"]})
    return {name: COLORS[i % len(COLORS)] for i, name in enumerate(names)}


def map_figure(result: dict, label: str, selected_unit: str | None = None) -> go.Figure:
    fc = result["map_result"]["solutions"][label]
    colors = measure_colors(result)
    boundary = shape(result["map_result"]["boundary"])
    fig = go.Figure()
    for name in sorted({f["properties"]["measures"] for f in fc["features"]}):
        features = [f for f in fc["features"] if f["properties"]["measures"] == name]
        color = colors[name]
        fig.add_trace(go.Choroplethmap(
            geojson={"type": "FeatureCollection", "features": features},
            featureidkey="properties.unit_id", locations=[f["properties"]["unit_id"] for f in features],
            z=[1] * len(features), colorscale=[[0, color], [1, color]], showscale=False,
            marker_opacity=0.8, marker_line_color="white", marker_line_width=2,
            name=name, showlegend=True,
            customdata=[[f["properties"][k] for k in ("unit_id", "risk", "safety", "lifecycle_cost", "resilience_change")]
                        for f in features],
            hovertemplate="<b>%{customdata[0]}</b><br>" + html.escape(name) +
                          "<br>风险：%{customdata[1]}级<br>安全：%{customdata[2]:.1f}分" +
                          "<br>成本：%{customdata[3]:.1f}万元<br>韧性变化：%{customdata[4]:+.1f}分<extra></extra>"))
    selected = [f for f in fc["features"] if f["properties"]["unit_id"] == selected_unit]
    for feature in selected:
        geom = shape(feature["geometry"])
        for polygon in ([geom] if geom.geom_type == "Polygon" else geom.geoms):
            x, y = polygon.exterior.xy
            fig.add_trace(go.Scattermap(lon=list(x), lat=list(y), mode="lines", line={"color": "#f2ae30", "width": 4},
                                        hoverinfo="skip", showlegend=False))
    for polygon in ([boundary] if boundary.geom_type == "Polygon" else boundary.geoms):
        x, y = polygon.exterior.xy
        fig.add_trace(go.Scattermap(lon=list(x), lat=list(y), mode="lines", name="村域边界",
                                   line={"color": "#33565e", "width": 2}, hoverinfo="skip", showlegend=False))
    points = [shape(f["geometry"]).representative_point() for f in fc["features"]]
    fig.add_trace(go.Scattermap(lon=[p.x for p in points], lat=[p.y for p in points], mode="markers+text",
                                marker={"size": 12, "color": "#173b45"},
                                text=[f["properties"]["unit_id"] for f in fc["features"]],
                                customdata=[[f["properties"]["unit_id"]] for f in fc["features"]],
                                textposition="top center", showlegend=False, hoverinfo="text"))
    bounds = boundary.bounds
    import math
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1e-5)
    zoom = max(1, min(18, math.log2(360 / span) - 0.35))
    fig.update_layout(height=560, margin={"l": 0, "r": 0, "t": 0, "b": 10},
                      map={"style": "white-bg", "center": {"lon": boundary.centroid.x, "lat": boundary.centroid.y}, "zoom": zoom},
                      paper_bgcolor="#f1f6f6", plot_bgcolor="#f1f6f6", font={"family": "Microsoft YaHei, sans-serif"},
                      legend={"orientation": "h", "y": -0.02, "x": 0}, clickmode="event+select",
                      uirevision=result["input_hash"])
    return fig


def radar_figure(result):
    fig = go.Figure()
    axes = list(OBJECTIVE_NAMES.values())
    for i, solution in enumerate(result["representative_solutions"]):
        scores = [solution["objectives"][k] for k in OBJECTIVE_NAMES]
        fig.add_trace(go.Scatterpolar(r=scores + scores[:1], theta=axes + axes[:1], name=solution["label"],
                                     line={"color": COLORS[i]}, fill="toself", opacity=0.55))
    fig.update_layout(height=350, margin={"t": 35, "l": 45, "r": 45, "b": 35},
                      polar={"radialaxis": {"range": [0, 100]}}, legend={"orientation": "h", "y": -0.15})
    return fig


def pareto_figure(result, label):
    frame = pd.DataFrame(result["pareto_front"])
    fig = go.Figure(go.Scatter(x=frame["economy"], y=frame["safety"], mode="markers",
                              marker={"size": 9, "color": frame["resilience"], "colorscale": "Teal", "showscale": True,
                                      "colorbar": {"title": "韧性"}},
                              customdata=frame[["plan_id", "sustainability", "total_cost"]].values.tolist(),
                              hovertemplate="%{customdata[0]}<br>经济性%{x:.1f}<br>安全性%{y:.1f}"+
                              "<br>可持续性%{customdata[1]:.1f}<br>成本%{customdata[2]:.1f}万元<extra></extra>"))
    selected = next(s for s in result["representative_solutions"] if s["label"] == label)
    fig.add_trace(go.Scatter(x=[selected["objectives"]["economy"]], y=[selected["objectives"]["safety"]],
                             mode="markers", marker={"size": 16, "symbol": "star", "color": "#d17a34"}, name=label))
    fig.update_layout(height=350, xaxis_title="经济性 / 分（越大越好）", yaxis_title="安全性 / 分（越大越好）",
                      margin={"t": 20, "l": 30, "r": 20, "b": 30}, showlegend=False)
    return fig


def composition_figure(result, label):
    solution = next(s for s in result["representative_solutions"] if s["label"] == label)
    counts = {}
    for detail in solution["unit_details"]:
        for name in detail["measure_names"]:
            counts[name] = counts.get(name, 0) + 1
    fig = go.Figure(go.Bar(x=list(counts.values()), y=list(counts), orientation="h", marker_color="#197278"))
    fig.update_layout(height=270, margin={"t": 10, "l": 20, "r": 20, "b": 20}, xaxis_title="配置单元数", xaxis_dtick=1)
    return fig


def _svg_paths(result):
    metric = CRS.from_user_input(result["metrics"]["spatial"]["metric_crs"])
    source = CRS.from_epsg(4326)
    boundary = reproject(shape(result["map_result"]["boundary"]), source, metric)
    x0, y0, x1, y1 = boundary.bounds
    scale = min(720 / (x1 - x0), 560 / (y1 - y0))

    def path(geometry):
        geom = reproject(shape(geometry), source, metric)
        output = []
        for polygon in ([geom] if geom.geom_type == "Polygon" else geom.geoms):
            for ring in [polygon.exterior, *polygon.interiors]:
                points = [(40 + (x - x0) * scale, 30 + (y1 - y) * scale) for x, y in ring.coords]
                output.append("M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in points) + "Z")
        centroid = geom.representative_point()
        return {"d": " ".join(output), "x": 40 + (centroid.x - x0) * scale, "y": 30 + (y1 - centroid.y) * scale}
    return path(result["map_result"]["boundary"]), {f["properties"]["unit_id"]: path(f["geometry"])
                                                        for f in result["governance_units"]["features"]}


def html_report(result: dict) -> str:
    """独立SVG/JS成果；所有文字使用textContent，输入数据不能注入HTML。"""
    boundary, paths = _svg_paths(result)
    data = {"summary": result["summary"], "solutions": result["representative_solutions"],
            "maps": result["map_result"]["solutions"], "colors": measure_colors(result),
            "boundary": boundary, "paths": paths, "input_hash": result["input_hash"]}
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>村域综合治理方案成果</title>
<style>*{box-sizing:border-box}body{margin:0;background:#eff4f3;color:#173b45;font:15px "Microsoft YaHei",sans-serif}
header{background:#173f49;color:white;padding:28px 5%}header p{color:#ccdfdc;margin-bottom:0}main{max-width:1380px;margin:auto;padding:24px}
.notice{border-left:4px solid #d99b42;padding:12px 16px;background:#fff9e9;margin-bottom:20px}select{padding:10px;border:1px solid #a6bcb9;border-radius:6px;font:inherit}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0}.card{background:white;border-radius:8px;padding:18px;border:1px solid #dde7e5}.value{font-size:25px;font-weight:700;margin-top:9px}
.layout{display:grid;grid-template-columns:2.2fr 1fr;gap:18px}.map{background:#f8fbfa;border-radius:8px;width:100%;border:1px solid #dde7e5}.unit{cursor:pointer;stroke:white;stroke-width:2}.unit:hover,.unit.active{stroke:#e6ac39;stroke-width:5}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0}.swatch{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:6px}
dt{color:#65817f;font-size:13px;margin-top:14px}dd{margin:4px 0;font-weight:600;line-height:1.5}h2{font-size:19px;margin-top:0}footer{color:#78918d;font-size:12px;overflow-wrap:anywhere;margin-top:20px}
@media(max-width:850px){.layout{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}main{padding:14px}}</style></head>
<body><header><h1>村域综合治理方案成果</h1><p id="village"></p></header><main><div class="notice" id="notice"></div>
<label for="scheme">代表方案　</label><select id="scheme" aria-label="代表方案"></select><div class="kpis" id="kpis"></div>
<div class="layout"><section><svg id="map" class="map" viewBox="0 0 800 640" aria-label="村域治理方案地图"></svg><div id="legend" class="legend"></div><p>点击治理单元查看措施与评价明细。图中为空间成果边界，未加载在线影像。</p></section>
<aside class="card"><h2 id="unitTitle">治理单元详情</h2><dl id="details"></dl></aside></div>
<section class="card"><h2>推荐依据</h2><p id="basis"></p><p id="paths"></p><p>完整非劣方案、四目标和数据来源见同目录 result.json。所有数值来自同一次计算快照。</p></section><footer id="trace"></footer></main>
<script>const D=__DATA__;const ns="http://www.w3.org/2000/svg";let unitId=Object.keys(D.paths)[0];
const el=id=>document.getElementById(id);const put=(id,text)=>el(id).textContent=text;
put('village',D.summary.village_name+' · '+D.summary.governance_unit_count+' 个治理单元');
put('notice',D.summary.notices.join(' '));put('trace','计算输入标识：'+D.input_hash);
D.solutions.forEach(s=>{const o=document.createElement('option');o.value=s.label;o.textContent=s.label;el('scheme').append(o)});
function draw(){const s=D.solutions.find(s=>s.label===el('scheme').value),fs=D.maps[s.label].features;
el('kpis').replaceChildren();[['全寿命成本 / 万元',s.total_cost],['安全性 / 分',s.objectives.safety],['经济性 / 分',s.objectives.economy],['可持续性 / 分',s.objectives.sustainability],['韧性 / 分',s.objectives.resilience]].forEach(([k,v])=>{let card=document.createElement('div');card.className='card';card.textContent=k;let value=document.createElement('div');value.className='value';value.textContent=v.toFixed(1);card.append(value);el('kpis').append(card)});
el('map').replaceChildren();let b=document.createElementNS(ns,'path');b.setAttribute('d',D.boundary.d);b.setAttribute('fill','#e7efeb');b.setAttribute('stroke','#6d8982');el('map').append(b);
fs.forEach(f=>{const p=f.properties,g=D.paths[p.unit_id];let path=document.createElementNS(ns,'path');path.setAttribute('d',g.d);path.setAttribute('fill-rule','evenodd');path.setAttribute('fill',D.colors[p.measures]);path.setAttribute('class','unit'+(p.unit_id===unitId?' active':''));path.setAttribute('data-unit',p.unit_id);path.setAttribute('tabindex','0');path.setAttribute('aria-label',p.unit_id+' '+p.measures);path.addEventListener('click',()=>{unitId=p.unit_id;draw()});path.addEventListener('keydown',e=>{if(e.key==='Enter'){unitId=p.unit_id;draw()}});el('map').append(path);let text=document.createElementNS(ns,'text');text.setAttribute('x',g.x);text.setAttribute('y',g.y);text.setAttribute('text-anchor','middle');text.setAttribute('fill','white');text.setAttribute('pointer-events','none');text.textContent=p.unit_id;el('map').append(text)});
el('legend').replaceChildren();[...new Set(fs.map(f=>f.properties.measures))].forEach(name=>{let item=document.createElement('span'),swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=D.colors[name];item.append(swatch,document.createTextNode(name));el('legend').append(item)});
const p=fs.find(f=>f.properties.unit_id===unitId).properties;put('unitTitle',p.unit_id+' 治理单元详情');el('details').replaceChildren();
[['推荐措施',p.measures],['治理面积',(p.area_m2/10000).toFixed(2)+' 公顷'],['治理前风险',p.risk+' 级'],['安全改善',p.safety_before.toFixed(1)+' → '+p.safety.toFixed(1)+' 分'],['全寿命成本',p.lifecycle_cost.toFixed(2)+' 万元'],['韧性变化',p.resilience_before.toFixed(1)+' → '+p.resilience.toFixed(1)+' 分'],['纳入依据',p.selection_reasons.join('；')],['复核事项',p.expert_review]].forEach(([k,v])=>{let dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=v;el('details').append(dt,dd)});
put('basis',s.basis);put('paths','关键路径完整覆盖率：'+s.path_coverage.toFixed(1)+'%。区域协同已参与候选方案评价和非劣比较。')}
el('scheme').addEventListener('change',draw);draw();</script></body></html>'''
    return template.replace("__DATA__", encoded)


def export_files(result: dict) -> dict[str, bytes]:
    files = {}
    for name, value in (("result.json", result), ("governance_units.geojson", result["governance_units"]),
                        ("priority_areas.geojson", result["priority_areas"])):
        files[name] = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    files["pareto_front.csv"] = pd.DataFrame(result["pareto_front"]).to_csv(index=False).encode("utf-8-sig")
    if result["status"] == "completed":
        files["map.html"] = html_report(result).encode("utf-8")
        from .geographic_map import result_map_html
        from .engineering_symbols import SYMBOL_NAMES, symbol_svg
        files["spatial_decision_map.html"] = result_map_html(result).encode("utf-8")
        if result.get("evaluation_areas"):
            files["evaluation_areas.geojson"] = json.dumps(result["evaluation_areas"], ensure_ascii=False).encode("utf-8")
        for mid in SYMBOL_NAMES:
            files[f"symbols/{mid}.svg"] = symbol_svg(mid).encode("utf-8")
        for label, fc in result["map_result"].get("measure_placements", {}).items():
            files[f"measures_{label}.geojson"] = json.dumps(fc, ensure_ascii=False, indent=2).encode("utf-8")
        for label, fc in result["map_result"]["solutions"].items():
            files[f"map_{label}.geojson"] = json.dumps(fc, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        files["screening.csv"] = pd.DataFrame(result["screening"]).to_csv(index=False).encode("utf-8-sig")
        files["iteration_history.csv"] = pd.DataFrame(result["metrics"]["iteration_history"]).to_csv(index=False).encode("utf-8-sig")
    return files


def export_zip(result: dict) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, value in export_files(result).items():
            archive.writestr(name, value)
    return output.getvalue()
