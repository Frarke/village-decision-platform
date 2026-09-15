"""可由宿主地图直接复用的工程措施SVG符号及单元内配置示意点。"""
from pyproj import CRS
from shapely.geometry import box, mapping, shape

from .data_adapter import reproject

SYMBOL_NAMES = {"M01":"截排水工程", "M02":"灌草种植", "M03":"植被混凝土", "M04":"格构锚固",
                "M05":"挡墙工程", "M06":"抗滑桩", "M07":"谷坊坝", "M08":"柔性拦挡结构"}
COLORS = {"M01":"#168bc1", "M02":"#39a65b", "M03":"#7ea954", "M04":"#da9b37",
          "M05":"#d46a42", "M06":"#8f70b7", "M07":"#297e96", "M08":"#a15c86"}
GROUND = '<path d="M6 57 45 38 90 56 50 77Z" fill="#b9cdae" stroke="#63775c" stroke-width="1.5"/>'
BODIES = {
    "M01": '<path d="m13 48 30-16 39 19-11 6-28-14-22 12Z" fill="#d6e7ed" stroke="#426780" stroke-width="2"/><path d="m21 48 22-11 31 16" fill="none" stroke="#119dde" stroke-width="5"/><path d="m68 46 9 7-11 2" fill="none" stroke="#11618b" stroke-width="3"/>',
    "M02": '<path d="M22 58V30m23 23V22m22 35V31" stroke="#4f6230" stroke-width="4"/><path d="m10 38 12-19 12 19Zm23-8 12-20 12 20Zm21 9 13-20 13 20Z" fill="#388d48" stroke="#236235" stroke-width="2"/><path d="m25 63 4-10 3 8m18 7 4-11 4 8" fill="none" stroke="#237b39" stroke-width="3"/>',
    "M03": '<path d="m15 61 19-39 42 13 10 29Z" fill="#adb6a0" stroke="#667d61" stroke-width="2"/><path d="m29 53 8-19m8 23 7-18m11 20 6-17" stroke="#3f984a" stroke-width="8" stroke-dasharray="4 3"/>',
    "M04": '<path d="m10 63 28-46 43 17 7 31Z" fill="#b7aa91" stroke="#746b56"/><path d="m22 51 57 8M29 37l54 9M38 22l45 11M34 61l17-36M54 64l12-33" stroke="#f1d887" stroke-width="5"/><path d="m31 45 9 10m14-14 9 10m4-1 9 10" stroke="#705333" stroke-width="3"/>',
    "M05": '<path d="m22 30 13-7 40 17-12 7Z" fill="#d9bd96" stroke="#735d43"/><path d="m22 30 41 17 6 20-53-22Z" fill="#b59977" stroke="#735d43" stroke-width="2"/><path d="m63 47 12-7 5 18-11 9Z" fill="#8b7359" stroke="#735d43"/><path d="m21 38 44 18m-32-19-2 6m18-1-1 7m-10-5-1 6m21-1v8" stroke="#6f604e" stroke-width="1.5"/>',
    "M06": '<g fill="#b5a2c7" stroke="#594871" stroke-width="2"><path d="m17 29 9-4 8 4-1 35-15-6Z"/><path d="m39 37 9-4 8 4-1 34-15-6Z"/><path d="m62 43 9-4 8 4-1 34-15-6Z"/></g><path d="m17 29 9 4 8-4m-8 4v28m13-24 9 4 8-4m-8 4v28m14-26 9 4 8-4m-8 4v27" fill="none" stroke="#594871"/>',
    "M07": '<path d="m12 46 28 16 43-22M28 37l23 13 32-15" fill="none" stroke="#148cc0" stroke-width="7"/><path d="m33 28 12-7 27 25-13 7Z" fill="#d6c3a4" stroke="#6e6758"/><path d="m33 28 26 25 4 15-33-31Z" fill="#a39b86" stroke="#6e6758" stroke-width="2"/><path d="m59 53 13-7 2 13-11 9Z" fill="#7a8177" stroke="#6e6758"/>',
    "M08": '<path d="M19 60V25m26 47V37m28 40V46" stroke="#704a69" stroke-width="4"/><path d="m19 29 54 23M19 40l54 23M19 51l54 23m-42-40v27m28-15v27" stroke="#977c99" stroke-width="2"/><path d="m19 29 26 32m0-20L19 51m26-10 28 33m0-22L45 63" stroke="#a087a0" fill="none"/>',
}


def symbol_svg(measure_id: str) -> str:
    body = BODIES.get(measure_id, '<path d="M25 27h45v35H25Z" fill="#78939c"/><text x="48" y="52" text-anchor="middle" fill="white" font-size="24">工</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 84" role="img">' + GROUND + body + '</svg>'


def measure_placements(solutions: dict, metric_crs: str) -> dict:
    source, metric = CRS.from_epsg(4326), CRS.from_user_input(metric_crs)
    output = {}
    for label, collection in solutions.items():
        points = []
        for feature in collection["features"]:
            props = feature["properties"]
            ids = [mid for mid in props["measure_ids"].split("+") if mid != "NONE"]
            geom = reproject(shape(feature["geometry"]), source, metric)
            x0, y0, x1, y1 = geom.bounds
            for i, mid in enumerate(ids):
                slice_geom = geom.intersection(box(x0+(x1-x0)*i/len(ids), y0-1,
                                                   x0+(x1-x0)*(i+1)/len(ids), y1+1))
                point = (slice_geom if not slice_geom.is_empty else geom).representative_point()
                points.append({"type":"Feature", "geometry":mapping(reproject(point,metric,source)),
                    "properties":{"unit_id":props["unit_id"],"measure_id":mid,
                                  "measure_name":props["measure_names"][i], "symbol_key":mid,
                                  "placement_basis":"单元内部代表点，仅表达技术配置归属", "is_design_location":False,
                                  "quantity_meaning":"图标个数不代表工程数量", "data_status":props["data_status"]}})
        output[label] = {"type":"FeatureCollection", "features":points}
    return output
