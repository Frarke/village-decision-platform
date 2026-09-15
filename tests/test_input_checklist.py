import json
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from api import app
from model.input_checklist import template_files, encode_files, decode_files, validate_files
from model.input_package import build_input_package
from model.service import load_geographic_sample
from io import BytesIO
from zipfile import ZipFile


@pytest.fixture(scope="module")
def files():
    return template_files()


def test_builtins_are_complete_and_compute_without_upload(files):
    client = TestClient(app)
    defaults = client.get("/input-defaults").json()
    assert len(defaults["catalog"]) == 11
    check = client.post("/validate-files", json={"files": defaults["files"]}).json()
    assert check["valid"] and all(r["valid"] for r in check["items"])
    response = client.post("/optimize-files", json={"files": defaults["files"]})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["input_hash"] == check["summary"]["input_hash"]
    assert len(result["governance_units"]["features"]) == 4
    assert result["summary"]["baseline_objectives"]["safety"] < result["recommended_solution"]["objectives"]["safety"]


def test_missing_required_file_blocks_and_restoration_recovers(files):
    missing = dict(files)
    del missing["risk.tif"]
    report, request = validate_files(missing)
    assert request is None and not report["valid"]
    assert not next(r for r in report["items"] if r["key"] == "risk.tif")["present"]
    assert validate_files(files)[0]["valid"]


@pytest.mark.parametrize("name,data", [
    ("boundary.geojson", b"[]"), ("risk.tif", b"not a tiff"),
    ("engineering_profiles.csv", b"profile_id,slope_height\n1,NaN"),
    ("project.json", b'{"schema_version":"1.0"}'),
    ("decision_context.geojson", b'{"type":"Point"}'),
])
def test_bad_replacement_is_retained_as_error_not_default(files, name, data):
    changed = {**files, name: data}
    report, request = validate_files(changed)
    assert request is None and not report["valid"]
    row = next(r for r in report["items"] if r["key"] == name)
    assert row["present"] and not row["valid"]


def test_optional_layers_can_be_removed_without_stale_manifest(files):
    missing = {k: v for k, v in files.items() if k not in {"susceptibility.tif", "vulnerability.tif"}}
    report, request = validate_files(missing)
    assert report["valid"]
    assert "susceptibility" not in request["spatial_layers"]["layers"]


def test_actual_replacement_budget_and_raster_affect_model(files):
    request = load_geographic_sample()
    request["spatial_layers"]["layers"]["slope_deg"][20][20] = 23
    with ZipFile(BytesIO(build_input_package(request))) as z:
        changed = {**files, "slope_deg.tif": z.read("slope_deg.tif")}
    report, payload = validate_files(changed, total_budget=400)
    original, _ = validate_files(files)
    assert report["valid"]
    assert payload["spatial_layers"]["layers"]["slope_deg"][20][20] == 23
    assert payload["configuration"]["solver"]["total_budget"] == 400
    assert report["summary"]["input_hash"] != original["summary"]["input_hash"]
    client = TestClient(app)
    response = client.post("/optimize-files", json={"files": encode_files(changed), "total_budget": 400})
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["input_hash"] == report["summary"]["input_hash"]
    assert all(s["total_cost"] <= 400 for s in result["representative_solutions"])


def test_cross_file_bindings_are_checked_before_calculation(files):
    context = json.loads(files["decision_context.geojson"])
    context["features"][0]["geometry"]["coordinates"] = [0, 0]
    report, request = validate_files({**files, "decision_context.geojson": json.dumps(context).encode()})
    assert request is None and "对应0" in report["errors"][0]


def test_renamed_same_extension_accepted_wrong_extension_and_base64_rejected(files):
    encoded = encode_files(files)
    encoded["risk.tif"]["name"] = "当地风险.tif"
    decoded, errors = decode_files(encoded)
    assert decoded["risk.tif"] == files["risk.tif"] and not errors
    encoded["risk.tif"]["name"] = "risk.png"
    encoded["haz.tif"]["content"] = "!bad!"
    decoded, errors = decode_files(encoded)
    assert set(errors) == {"risk.tif", "haz.tif"}
    assert not validate_files(decoded, errors)[0]["valid"]


def test_real_project_cannot_silently_reuse_mock_defaults(files):
    project = json.loads(files["project.json"])
    project["village"]["data_status"] = "REAL"
    report, _ = validate_files({**files, "project.json": json.dumps(project).encode()})
    assert not report["valid"] and "内置模拟文件" in report["errors"][0]


def test_invalid_file_set_api_cannot_compute(files):
    client = TestClient(app)
    encoded = encode_files(files)
    encoded.pop("risk.tif")
    assert client.post("/optimize-files", json={"files": encoded}).status_code == 422
    assert client.post("/validate-files", json={"files": {"unknown": {}}}).status_code == 422
    assert client.post("/validate-files", json={"files": encode_files(files), "total_budget": -1}).json()["valid"] is False
