from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app
from model.nsga2 import solve_nsga2
from model.pareto import OBJECTIVE_COLUMNS, dominates
from model.regional import evaluate_village_scheme, normalized_risk_weights
from model.result_exporter import export_zip, map_figure
from model.schemas import SolverConfig
from model.service import (input_hash, load_sample_request, model_context, optimize_request, prepare_request)
from model.solver import configuration_hash, solve_exact
from model.spatial import evaluate_path_coverage
from model.validation import compare_exact_nsga
from model.workflow import optimization_hash


@pytest.fixture(scope="module")
def business():
    request = load_sample_request()
    prepared, units, measures, config, topology, _, _ = model_context(request)
    exact = solve_exact(units, measures, config, topology=topology)
    nsga = solve_nsga2(units, measures, config, topology=topology)
    return request, prepared, units, measures, config, topology, exact, nsga


@pytest.fixture(scope="module")
def response():
    return optimize_request(load_sample_request())


def test_exact_and_nsga_same_encoding_objectives_constraints(business):
    _, _, _, _, _, _, exact, nsga = business
    expected = exact.all_feasible.set_index("configuration_summary")
    assert len(nsga.all_feasible) > 0
    for row in nsga.all_feasible.to_dict("records"):
        e = expected.loc[row["configuration_summary"]]
        assert [row[k] for k in OBJECTIVE_COLUMNS] == e[OBJECTIVE_COLUMNS].tolist()
        assert json.loads(row["constraint_status"]) == json.loads(e["constraint_status"])
        assert json.loads(row["normalized_risk_weights"]) == json.loads(e["normalized_risk_weights"])
    comparison = compare_exact_nsga(exact, nsga)
    assert comparison["被精确Pareto集支配比例"] == 0
    assert comparison["精确Pareto集完全方案覆盖率"] >= .9


def test_exact_pareto_is_not_dominated_and_nsga_reproducible(business):
    _, _, units, measures, config, topology, exact, nsga = business
    feasible = exact.all_feasible[OBJECTIVE_COLUMNS].to_numpy()
    for point in exact.pareto[OBJECTIVE_COLUMNS].to_numpy():
        assert not any(dominates(other, point) for other in feasible)
    repeat = solve_nsga2(units, measures, config, topology=topology)
    pd.testing.assert_frame_equal(repeat.pareto, nsga.pareto)
    pd.testing.assert_frame_equal(repeat.iteration_history, nsga.iteration_history)
    assert repeat.iteration_history.generation.tolist() == list(range(config.nsga_generations + 1))


def test_illegal_direct_scheme_is_infeasible_without_mutation(business):
    _, _, units, measures, config, topology, _, _ = business
    wall = next(m for m in measures if m.measure_id == "M05")
    selected = [(wall, wall), (), (), ()]
    original = deepcopy(selected)
    evaluated = evaluate_village_scheme(units, selected, config, topology)
    assert not evaluated["feasible"]
    assert any("未准入" in r for r in evaluated["constraint_reasons"])
    assert any("重复" in r for r in evaluated["constraint_reasons"])
    assert selected == original


def test_topology_is_in_objectives_local_counts_and_all_hashes(business):
    request, _, units, measures, config, topology, exact, _ = business
    by_id = {m.measure_id: m for m in measures}
    details = exact.detail_lookup[str(exact.representatives.iloc[0].plan_id)]
    selected = [tuple(by_id[x] for x in d["measure_ids"].split("+") if x != "NONE") for d in details]
    one = evaluate_village_scheme(units, selected, config, topology)
    changed_units = [replace(u, key_paths=999) for u in units]
    assert evaluate_village_scheme(changed_units, selected, config, topology) == one
    assert one["path_details"] == evaluate_path_coverage(topology, one["unit_details"])["path_details"]
    empty = {"paths": []}
    two = evaluate_village_scheme(units, selected, config, empty)
    assert one["regional_objectives"]["resilience"] != two["regional_objectives"]["resilience"]
    assert configuration_hash(units, measures, config, "exact", topology) != configuration_hash(units, measures, config, "exact", empty)
    assert optimization_hash(units, measures, config, "exact", topology) != optimization_hash(units, measures, config, "exact", empty)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(unit_risk_weights={}), "显式"),
    (lambda r: r.update(unit_risk_weights={f"U{i:02d}": 0 for i in range(1, 5)}), "大于零"),
    (lambda r: r["engineering_profiles"]["1"].pop("anchor_condition"), "缺少"),
    (lambda r: r["engineering_profiles"]["1"].update(anchor_condition="False"), "布尔"),
    (lambda r: r.pop("topology"), "topology"),
    (lambda r: r["topology"].update(paths=[]), "confirmed_empty"),
    (lambda r: r["topology"]["paths"][0].update(source_unit="ghost"), "不存在"),
    (lambda r: r["configuration"]["solver"].update(nsga_mutation_probability=2), "概率"),
    (lambda r: r["configuration"]["solver"].update(nsga_generations=2.5), "整数"),
    (lambda r: r["configuration"]["solver"].update(total_budget=0), "预算"),
])
def test_bad_data_never_falls_back_to_mock(mutation, match):
    request = load_sample_request()
    mutation(request)
    with pytest.raises(ValueError, match=match):
        optimize_request(request)


def test_results_include_all_representatives_geometry_baselines_and_exports(response):
    assert [s["label"] for s in response["representative_solutions"]] == ["综合均衡", "安全优先", "经济优先", "韧性优先", "可持续优先"]
    assert response["recommended_solution"] == response["representative_solutions"][0]
    assert len({tuple(tuple(c) for c in s["configuration"]) for s in response["representative_solutions"]}) > 1
    for solution in response["representative_solutions"]:
        features = response["map_result"]["solutions"][solution["label"]]["features"]
        assert sum(f["properties"]["lifecycle_cost"] for f in features) == pytest.approx(solution["total_cost"])
        assert all(f["properties"]["data_status"] == "MOCK" for f in features)
        assert all("available_measures" in f["properties"] for f in features)
        assert all(f["properties"]["key_paths"] == f["properties"]["engineering_attributes"]["key_paths"] for f in features)
        assert all(solution["constraint_status"].values())
        fig = map_figure(response, solution["label"], "U02")
        assert len(fig.data) >= 3
    with ZipFile(BytesIO(export_zip(response))) as archive:
        assert {"map.html", "result.json", "governance_units.geojson", "pareto_front.csv", "iteration_history.csv"} <= set(archive.namelist())
        assert json.loads(archive.read("result.json"))["input_hash"] == response["input_hash"]
        content = archive.read("map.html").decode()
        assert "<svg" in content and "addEventListener('change'" in content
        assert "https://" not in content  # SVG规范命名空间为http，不加载任何外部资产。


def test_api_prepare_optimize_and_errors(response):
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    request = load_sample_request()
    assert client.post("/prepare", json=request).json()["required_risk_weight_ids"] == ["U01", "U02", "U03", "U04"]
    reply = client.post("/optimize", json=request)
    assert reply.status_code == 200, reply.text
    assert reply.json()["pareto_front"] == response["pareto_front"]
    request["configuration"]["solver"]["total_budget"] = -1
    assert client.post("/optimize", json=request).status_code == 422
    request["server_file_path"] = "private"
    assert client.post("/optimize", json=request).status_code == 422


def test_empty_region_returns_no_invented_solutions():
    request = load_sample_request()
    request["configuration"]["spatial"]["min_area_m2"] = 100_000
    result = optimize_request(request)
    assert result["status"] == "no_governance_units"
    assert result["recommended_solution"] is None
    assert result["pareto_front"] == []


def test_large_explicit_risk_weights_normalize_without_overflow(business):
    units = [replace(u, risk_weight=1e308) for u in business[2]]
    assert normalized_risk_weights(units) == {u.unit_id: .25 for u in units}


def test_invalid_unscaled_baseline_score_is_rejected():
    request = load_sample_request()
    request["configuration"]["solver"]["evaluation_parameters"] = {"untreated_sustainability": 500}
    with pytest.raises(ValueError, match="0～100"):
        optimize_request(request)
