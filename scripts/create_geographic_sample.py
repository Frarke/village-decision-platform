"""在已核验真实地点生成连续模拟评价场；治理边界由模型另行计算。"""
from pathlib import Path
import json
import sys

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.input_package import build_input_package
from model.schemas import load_units
from model.service import AUTOMATIC_FIELDS, prepare_request


def main():
    # OSM node8199467914 / Nominatim实查地点；此处不是行政村法定边界。
    longitude, latitude = 116.6346480, 25.6390104
    forward = Transformer.from_crs(4326, 32650, always_xy=True)
    inverse = Transformer.from_crs(32650, 4326, always_xy=True)
    cx, cy = forward.transform(longitude, latitude)
    n, resolution = 48, 40
    x0, y0 = cx - 960, cy + 960
    rows, cols = np.mgrid[:n, :n]
    centers = [(12, 12), (13, 35), (34, 13), (35, 35)]
    fields = [np.exp(-(((rows-r)/5.2)**2 + ((cols-c)/4.5)**2)/2) for r, c in centers]
    strength = np.maximum.reduce(fields)
    strength = np.clip(strength * (1 + .10*np.sin(rows*.8)*np.cos(cols*.6)), 0, 1)
    profile = 1 + (cols >= 24).astype(int) + 2*(rows >= 24).astype(int)
    risk = np.clip(np.floor(1 + 3.6*strength), 1, 4)
    risk = np.where(profile == 2, np.minimum(risk, 2), risk)
    risk = np.where(profile == 4, np.minimum(risk, 3), risk)
    slope_base = np.choose(profile-1, [39., 19., 27., 31.])
    layers = {
        "risk": risk.astype(int).tolist(), "haz": risk.astype(int).tolist(),
        "distance_m": np.round(28 + 540*(1-strength), 2).tolist(),
        "slope_deg": np.round(10 + (slope_base-10)*strength + 2*np.sin(cols/5)**2, 2).tolist(),
        "susceptibility": np.round(np.clip(.12 + .83*strength + .03*np.sin(rows/8), 0, 1), 4).tolist(),
        "vulnerability": np.round(np.clip(.18 + .66*strength + .07*np.cos(cols/7), 0, 1), 4).tolist(),
        "profile_id": profile.tolist(),
    }
    boundary = Polygon([(cx+x, cy+y) for x, y in [(-900,-720),(-945,-150),(-830,670),(-300,910),
                         (450,860),(880,600),(930,-260),(620,-860),(-150,-900)]])
    original = load_units(ROOT / "data/default_units.csv")
    profiles = {str(i): {**{k:v for k,v in u.to_dict().items() if k not in AUTOMATIC_FIELDS},
                         "evidence_level": "MOCK：模拟工程调查，非培田村勘察成果"} for i,u in enumerate(original,1)}
    coordinates = [list(inverse.transform(x0+(c+.5)*resolution, y0-(r+.5)*resolution)) for r,c in centers]
    features = [{"type":"Feature", "geometry":{"type":"Point","coordinates":point},
                 "properties":{"record_id":f"W{i}","kind":"weight","risk_weight":original[i-1].risk_weight}}
                for i,point in enumerate(coordinates,1)]
    for index, stages in enumerate(((0,2,1),(2,0,3),(3,2,1)),1):
        features.append({"type":"Feature", "geometry":{"type":"LineString","coordinates":[coordinates[i] for i in stages]},
                         "properties":{"record_id":f"P{index}","kind":"path","name":f"模拟风险传递路径{index}","key":True}})
    request = {
        "schema_version":"1.0",
        "village":{"id":"PEITIAN-MOCK-01","name":"培田村周边研究范围（模拟评价）","data_status":"MOCK",
                   "source":"真实地点：OpenStreetMap node8199467914；研究范围、评价栅格、调查和路径均为模拟，不是行政村法定边界或当地灾害调查结果。",
                   "location_source":"https://www.openstreetmap.org/node/8199467914",
                   "crs":"EPSG:4326","boundary":mapping(transform(inverse.transform,boundary))},
        "spatial_layers":{"kind":"raster_grid","crs":"EPSG:32650","transform":[40,0,x0,0,-40,y0],"layers":layers},
        "engineering_profiles":profiles,
        "spatial_bindings":{"type":"FeatureCollection","data_status":"MOCK","source":"显式模拟权重点和三环节路径坐标，非地形自动推断", "features":features},
        "configuration":{"method":"nsga2","spatial":{"min_area_m2":1600,"merge_distance_m":0},
                         "solver":{"total_budget":560,"nsga_population_size":60,"nsga_generations":25,"random_seed":42}},
    }
    prepared = prepare_request(request)
    folder = ROOT / "data/v1_1"
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"peitian_request.json").write_text(json.dumps(request,ensure_ascii=False,indent=2),"utf-8")
    (folder/"peitian_input_package.zip").write_bytes(build_input_package(request))
    (folder/"location_source.json").write_text(json.dumps({"provider":"OpenStreetMap / Nominatim", "osm_node":8199467914,
        "name":"培田村, 龙岩市, 福建省, 中国", "longitude":longitude,"latitude":latitude,
        "license":"OpenStreetMap contributors, ODbL 1.0", "boundary_status":"演示研究范围，非法定行政村界"},ensure_ascii=False,indent=2),"utf-8")
    print(json.dumps(prepared["spatial_metrics"],ensure_ascii=False,indent=2))
    print('生成单元：', prepared['required_risk_weight_ids'])


if __name__ == '__main__':
    main()
