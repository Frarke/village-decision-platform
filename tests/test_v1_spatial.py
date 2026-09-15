from copy import deepcopy
import base64
import json

import numpy as np
import pytest
from rasterio.io import MemoryFile
from affine import Affine
from shapely.geometry import box, mapping, shape

from model.data_adapter import geotiff_grid, grid_features, read_spatial_file
from model.governance import SpatialConfig, build_governance, priority_reasons
from model.service import load_sample_request, prepare_request


@pytest.fixture
def request_data():
    return load_sample_request()


def small_vector(boxes, profiles=None):
    profiles = profiles or ["1"] * len(boxes)
    village = {"crs": "EPSG:32650", "boundary": mapping(box(408700, 2775000, 410000, 2776800)), "data_status": "MOCK"}
    features = [{"type": "Feature", "geometry": mapping(g), "properties": {
        "feature_id": f"F{i}", "profile_id": pid, "risk": 3, "haz": 2, "distance_m": 200,
        "slope_deg": 20 + i * 10, "susceptibility": 0.2 + i * .1, "vulnerability": .5}}
        for i, (g, pid) in enumerate(zip(boxes, profiles))]
    return village, {"kind": "vector", "crs": "EPSG:32650", "features": features}


def test_sample_starts_from_cells_and_builds_four_valid_units(request_data):
    output = prepare_request(request_data)
    assert output["spatial_metrics"]["valid_features"] == 144
    assert output["spatial_metrics"]["priority_features"] == 24
    assert output["required_risk_weight_ids"] == ["U01", "U02", "U03", "U04"]
    for feature in output["governance_units"]["features"]:
        assert shape(feature["geometry"]).is_valid
        assert feature["properties"]["area_m2"] == pytest.approx(60_000)
        assert len(feature["properties"]["source_feature_ids"]) == 6


def test_report_threshold_boundaries_and_optional_layers_not_hidden_rules():
    p = {"risk": 2, "distance_m": 100, "susceptibility": 1, "vulnerability": 1}
    assert priority_reasons(p, SpatialConfig()) == ["距承灾体≤100米"]
    p["distance_m"] = 101
    assert not priority_reasons(p, SpatialConfig())
    p["risk"] = 3
    assert priority_reasons(p, SpatialConfig()) == ["风险等级≥3"]


def test_connected_area_weighted_mean_and_conservative_slope():
    village, layer = small_vector([box(408800, 2775500, 408810, 2775510), box(408810, 2775500, 408830, 2775510)])
    layer["features"][0]["properties"]["feature_id"] = 0  # CSV/GIS常见数字编号，统一为文本溯源。
    output = build_governance(village, layer, SpatialConfig(min_area_m2=0))
    props = output["governance_units"]["features"][0]["properties"]
    assert len(output["governance_units"]["features"]) == 1
    assert props["area_m2"] == pytest.approx(300)
    assert props["mean_slope_deg"] == pytest.approx(80 / 3)
    assert props["slope_deg"] == 30
    assert props["mean_susceptibility"] == pytest.approx(0.8 / 3)
    assert props["source_feature_ids"] == ["0", "F1"]


def test_corner_contact_stays_separate_and_profiles_stop_merge():
    village, layer = small_vector([box(408800, 2775500, 408810, 2775510), box(408810, 2775510, 408820, 2775520)])
    assert len(build_governance(village, layer, SpatialConfig(min_area_m2=0))["governance_units"]["features"]) == 2
    layer["features"][1]["geometry"] = mapping(box(408810, 2775500, 408820, 2775510))
    layer["features"][1]["properties"]["profile_id"] = "2"
    assert len(build_governance(village, layer, SpatialConfig(min_area_m2=0))["governance_units"]["features"]) == 2


def test_merge_does_not_add_buffer_area_and_split_conserves_area():
    village, layer = small_vector([box(408800, 2775500, 408810, 2775510), box(408815, 2775500, 408825, 2775510)])
    out = build_governance(village, layer, SpatialConfig(min_area_m2=0, merge_distance_m=6))
    assert len(out["governance_units"]["features"]) == 1
    assert out["governance_units"]["features"][0]["properties"]["area_m2"] == pytest.approx(200)
    out = build_governance(village, layer, SpatialConfig(min_area_m2=0, merge_distance_m=6, max_unit_area_m2=50))
    assert sum(f["properties"]["area_m2"] for f in out["governance_units"]["features"]) == pytest.approx(200)
    assert all(f["properties"]["area_m2"] <= 50.00001 for f in out["governance_units"]["features"])


def test_small_patch_filter_reports_lost_area():
    village, layer = small_vector([box(408800, 2775500, 408805, 2775505)])
    out = build_governance(village, layer)
    assert not out["governance_units"]["features"]
    assert out["spatial_metrics"]["filtered_area_m2"] == pytest.approx(25)


def test_disconnected_parts_of_one_source_feature_form_separate_units():
    from shapely.geometry import MultiPolygon
    village, layer = small_vector([MultiPolygon([box(408800, 2775500, 408820, 2775520),
                                                 box(408850, 2775500, 408870, 2775520)])])
    out = build_governance(village, layer)
    assert len(out["governance_units"]["features"]) == 2
    assert all(f["properties"]["source_feature_ids"] == ["F0"] for f in out["governance_units"]["features"])


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["spatial_layers"].pop("crs"), "CRS"),
    (lambda r: r["spatial_layers"]["layers"]["risk"][0].pop(), "形状|shape|inhomogeneous|网格"),
    (lambda r: r["spatial_layers"]["layers"]["risk"][0].__setitem__(0, 2.5), "整数"),
    (lambda r: r["spatial_layers"]["layers"]["vulnerability"][0].__setitem__(0, 2), "范围"),
    (lambda r: r["spatial_layers"].update(kind="png"), "不支持"),
])
def test_invalid_spatial_input_rejected(request_data, mutation, match):
    mutation(request_data)
    with pytest.raises(ValueError, match=match):
        prepare_request(request_data)


def test_invalid_geometry_overlap_and_duplicate_id():
    village, layer = small_vector([box(408800, 2775500, 408820, 2775520)] * 2)
    with pytest.raises(ValueError, match="重叠"):
        build_governance(village, layer)
    layer["features"][1]["properties"]["feature_id"] = "F0"
    with pytest.raises(ValueError, match="唯一"):
        build_governance(village, layer)
    layer["features"] = layer["features"][:1]
    layer["features"][0]["geometry"] = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    with pytest.raises(ValueError, match="有效"):
        build_governance(village, layer)


def test_nodata_and_no_priority_area_are_explicit(request_data):
    request_data["spatial_layers"]["layers"]["risk"][0][0] = None
    out = prepare_request(request_data)
    assert out["spatial_metrics"]["nodata_cells"] == 1
    request_data["spatial_layers"]["layers"]["risk"] = [[1] * 12 for _ in range(12)]
    request_data["spatial_layers"]["layers"]["distance_m"] = [[400] * 12 for _ in range(12)]
    out = prepare_request(request_data)
    assert out["status"] == "no_governance_units"


def test_geotiff_vector_and_grid_produce_same_units(request_data, tmp_path):
    spec = request_data["spatial_layers"]
    expected = prepare_request(request_data)["governance_units"]
    fields = list(spec["layers"])
    arrays = np.asarray([spec["layers"][key] for key in fields], dtype="float64")
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", height=12, width=12, count=len(fields), dtype="float64",
                         crs=spec["crs"], transform=Affine(*spec["transform"]), nodata=-9999) as dataset:
            dataset.write(arrays)
        content = memory.read()
    request_data["spatial_layers"] = {"kind": "geotiff", "content_base64": base64.b64encode(content).decode(), "band_fields": fields}
    assert prepare_request(request_data)["governance_units"] == expected
    path = tmp_path / "evaluation.tif"
    path.write_bytes(content)
    assert read_spatial_file(path, band_fields=fields)["crs"] == spec["crs"]
    features, _ = grid_features(spec)
    request_data["spatial_layers"] = {"kind": "vector", "crs": spec["crs"], "features": features}
    assert prepare_request(request_data)["governance_units"] == expected
    # 输入顺序不能改变单元编号、原片区溯源和几何面积。
    request_data["spatial_layers"]["features"].reverse()
    actual = prepare_request(request_data)["governance_units"]["features"]
    assert [f["properties"] for f in actual] == [f["properties"] for f in expected["features"]]


def test_wgs84_conversion_holes_and_csv_wkt(tmp_path):
    import pandas as pd
    village, layer = small_vector([box(408800, 2775500, 408830, 2775530).difference(box(408810, 2775510, 408820, 2775520))])
    out = build_governance(village, layer)
    geom = shape(out["governance_units"]["features"][0]["geometry"])
    assert 116 < geom.centroid.x < 117 and 25 < geom.centroid.y < 26
    assert len(geom.interiors) == 1
    row = {**layer["features"][0]["properties"], "geometry_wkt": shape(layer["features"][0]["geometry"]).wkt}
    file = tmp_path / "evaluation.csv"
    pd.DataFrame([row]).to_csv(file, index=False)
    assert len(build_governance(village, read_spatial_file(file, crs="EPSG:32650"))["governance_units"]["features"]) == 1
