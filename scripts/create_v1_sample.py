"""生成有真实空间参考的模拟评价网格；不使用已有治理单元geometry作为输入。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyproj import Transformer
from shapely.geometry import box, mapping
from model.schemas import load_units
from model.service import AUTOMATIC_FIELDS


def main():
    units = load_units(ROOT / "data/default_units.csv")
    x, y = Transformer.from_crs(4326, 32650, always_xy=True).transform(116.095, 25.103)
    x, y = round(x / 100) * 100, round(y / 100) * 100
    n = 12
    fields = ("risk", "haz", "distance_m", "slope_deg", "susceptibility", "vulnerability", "profile_id")
    layers = {key: [[0 for _ in range(n)] for _ in range(n)] for key in fields}
    patches = {(r, c): i for i, (r0, c0) in enumerate(((7, 1), (7, 8), (1, 1), (1, 8)), 1)
               for r in range(r0, r0 + 3) for c in range(c0, c0 + 2)}
    for r in range(n):
        for c in range(n):
            pid = patches.get((r, c), 0)
            unit = units[pid - 1] if pid else None
            values = {"risk": unit.risk if unit else 1, "haz": unit.haz if unit else 1,
                      "distance_m": unit.distance_m if unit else 350,
                      "slope_deg": unit.slope_deg - (c % 2) if unit else 12,
                      "susceptibility": round((unit.risk / 6 + (r % 3) / 40) if unit else 0.15, 3),
                      "vulnerability": round((0.72 if pid in (1, 2) else 0.55) if unit else 0.2, 3),
                      "profile_id": pid}
            for key, value in values.items():
                layers[key][r][c] = value
    profiles = {}
    for index, unit in enumerate(units, 1):
        profiles[str(index)] = {k: v for k, v in unit.to_dict().items() if k not in AUTOMATIC_FIELDS}
        profiles[str(index)]["evidence_level"] = "MOCK：模拟调查属性，待标定"
    topology = json.loads((ROOT / "data/spatial_topology.json").read_text("utf-8"))
    topology.update({"source": "原四单元案例的人工路径关系，用于新空间输入闭环验证", "data_status": "MOCK"})
    request = {
        "schema_version": "1.0",
        "village": {"id": "MOCK-VILLAGE-01", "name": "村域综合整治示例研究区", "data_status": "MOCK",
                    "source": "程序构建的12×12空间评价网格及模拟调查属性；非现场研究成果",
                    "crs": "EPSG:32650", "boundary": mapping(box(x, y - 1200, x + 1200, y))},
        "spatial_layers": {"kind": "raster_grid", "crs": "EPSG:32650", "transform": [100, 0, x, 0, -100, y], "layers": layers},
        "engineering_profiles": profiles,
        "unit_risk_weights": {u.unit_id: u.risk_weight for u in units}, "topology": topology,
        "configuration": {"method": "nsga2", "spatial": {"min_area_m2": 100, "merge_distance_m": 0},
                          "solver": {"total_budget": 520, "nsga_population_size": 60, "nsga_generations": 25, "random_seed": 42}},
    }
    output = ROOT / "data/v1"
    output.mkdir(exist_ok=True)
    (output / "sample_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), "utf-8")
    print("已生成MOCK输入：144个评价像元，非预定义治理单元。")


if __name__ == "__main__":
    main()
