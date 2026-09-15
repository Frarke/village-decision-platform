"""真实底图上的输入溯源及工程符号地图；宿主可嵌入HTML或直接使用GeoJSON。"""
from pathlib import Path
import json

from .engineering_symbols import COLORS, SYMBOL_NAMES, symbol_svg, map_symbol_svg

ROOT = Path(__file__).resolve().parents[1]


def geographic_map_html(prepared: dict, request: dict, result: dict | None = None,
                        label: str | None = None, selected_unit: str | None = None, embedded=False) -> str:
    payload = {"boundary":prepared["boundary"], "evaluation":prepared.get("evaluation_areas", {"type":"FeatureCollection","features":[]}),
               "priority":prepared["priority_areas"], "units":prepared["governance_units"],
               "metrics":prepared.get("spatial_metrics", {}), "village":request["village"],
               "context":request.get("spatial_bindings") or {"type":"FeatureCollection","features":[]},
               "result":result, "label":label, "selected":selected_unit, "embedded":embedded,
               "symbols":{mid:symbol_svg(mid) for mid in SYMBOL_NAMES}, "names":SYMBOL_NAMES,"colors":COLORS}
    payload["glyphs"] = {mid: map_symbol_svg(mid) for mid in SYMBOL_NAMES}
    # 输入文字只以textContent或转义文本使用；防止上传字符串关闭script标签。
    data = json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")).replace("<","\\u003c")
    template = (ROOT/"assets/geographic_map.html").read_text("utf-8")
    return (template.replace("__LEAFLET_CSS__",(ROOT/"assets/vendor/leaflet/leaflet.css").read_text("utf-8"))
            .replace("__LEAFLET_JS__",(ROOT/"assets/vendor/leaflet/leaflet.js").read_text("utf-8"))
            .replace("__PAYLOAD__",data))


def result_map_html(result: dict, label=None, selected_unit=None, embedded=False) -> str:
    prepared = {"boundary":result["map_result"]["boundary"], "evaluation_areas":result.get("evaluation_areas"),
                "priority_areas":result["priority_areas"], "governance_units":result["governance_units"],
                "spatial_metrics":result["metrics"]["spatial"]}
    return geographic_map_html(prepared, result["provenance"], result, label, selected_unit, embedded)
