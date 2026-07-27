from __future__ import annotations

from pathlib import Path
import re
import json
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.constraints import check_plan_constraints
from model.evaluation import evaluate_unit
from model.i18n import chinese_frame, unmapped_columns
from model.schemas import SolverConfig, load_measures, load_units
from model.solver import configuration_hash, is_result_stale, result_to_json, solve_exact
from model.nsga2 import solve_nsga2
from model.spatial import coordination_metrics, load_geojson, load_topology, spatial_unit_table
from model.validation import validate_runs


class RoundTwoAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = load_units(ROOT / "data/default_units.csv")
        cls.measures = load_measures(ROOT / "data/default_measures.csv")
        cls.config = SolverConfig(nsga_population_size=50, nsga_generations=8, random_seed=42)
        cls.exact = solve_exact(cls.units, cls.measures, cls.config, ROOT / "data/assumptions.csv")
        cls.nsga = solve_nsga2(cls.units, cls.measures, cls.config, ROOT / "data/assumptions.csv")

    def test_16_all_export_fields_have_chinese_mapping(self):
        frames = [self.exact.tagged_units, self.exact.measure_library, self.exact.screening,
                  self.exact.all_feasible, self.exact.pareto, self.exact.representatives,
                  self.exact.representative_details]
        for frame in frames:
            self.assertEqual(unmapped_columns(frame), [])
            translated = chinese_frame(frame)
            self.assertFalse(any(re.fullmatch(r"[a-z][a-z0-9_]*", str(column)) for column in translated.columns))

    def test_17_example_geojson_and_unit_linkage(self):
        geojson = load_geojson(ROOT / "data/spatial_units.geojson")
        table = spatial_unit_table(geojson)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(set(table["unit_id"]), {unit.unit_id for unit in self.units})
        self.assertTrue(table[["longitude", "latitude"]].notna().all().all())

    def test_18_spatial_coordination_metrics(self):
        topology = load_topology(ROOT / "data/spatial_topology.json")
        plan_id = str(self.exact.representatives.iloc[0]["plan_id"])
        metrics = coordination_metrics(topology, self.exact.detail_lookup[plan_id])
        self.assertEqual(metrics["路径总数"], 3)
        self.assertGreaterEqual(metrics["路径治理覆盖率（%）"], 0)
        self.assertLessEqual(metrics["路径治理覆盖率（%）"], 100)

    def test_19_scenario_hash_changes_with_config(self):
        first = configuration_hash(self.units, self.measures, SolverConfig(total_budget=520), "exact_enumeration")
        second = configuration_hash(self.units, self.measures, SolverConfig(total_budget=350), "exact_enumeration")
        self.assertNotEqual(first, second)
        self.assertRegex(self.exact.run_metadata["scenario_id"], r"^SCN-\d{8}-\d{6}-[A-F0-9]{6}$")
        self.assertTrue(is_result_stale(first, second))
        self.assertFalse(is_result_stale(first, first))

    def test_20_nsga2_same_seed_is_reproducible(self):
        repeated = solve_nsga2(self.units, self.measures, self.config, ROOT / "data/assumptions.csv")
        pd.testing.assert_frame_equal(self.nsga.pareto, repeated.pareto)
        pd.testing.assert_frame_equal(self.nsga.iteration_history, repeated.iteration_history)

    def test_21_nsga2_outputs_satisfy_hard_constraints(self):
        for plan_id, details in self.nsga.detail_lookup.items():
            total = sum(float(item["lifecycle_cost"]) for item in details)
            valid, reasons = check_plan_constraints(self.units, details, total, self.config)
            self.assertTrue(valid, f"{plan_id}: {reasons}")

    def test_22_nsga2_and_exact_share_objective_evaluation(self):
        exact_by_configuration = self.exact.all_feasible.set_index("configuration_summary")
        common = self.nsga.all_feasible[self.nsga.all_feasible["configuration_summary"].isin(exact_by_configuration.index)]
        self.assertGreater(len(common), 0)
        for row in common.head(20).itertuples():
            exact = exact_by_configuration.loc[row.configuration_summary]
            self.assertAlmostEqual(row.safety, exact.safety, places=6)
            self.assertAlmostEqual(row.economy, exact.economy, places=6)
            self.assertAlmostEqual(row.sustainability, exact.sustainability, places=6)
            self.assertAlmostEqual(row.resilience, exact.resilience, places=6)

    def test_23_validation_metrics_are_bounded(self):
        validation = validate_runs(self.exact, [self.nsga])
        values = validation.metrics.set_index("验证指标")["本次结果"]
        for key in ["可行解比例", "精确Pareto集完全方案覆盖率", "目标空间近似覆盖率", "被精确Pareto集支配比例", "目标空间范围覆盖程度"]:
            self.assertGreaterEqual(float(values[key]), 0.0)
            self.assertLessEqual(float(values[key]), 1.0)

    def test_24_lifecycle_cost_penalizes_added_measure(self):
        drainage = next(item for item in self.measures if item.measure_name == "截排水工程")
        grid = next(item for item in self.measures if item.measure_name == "格构锚固")
        one = evaluate_unit(self.units[0], (grid,), self.config)
        two = evaluate_unit(self.units[0], (drainage, grid), self.config)
        self.assertGreater(two["lifecycle_cost"], one["lifecycle_cost"])
        self.assertGreaterEqual(two["economy"], 0.0)
        self.assertLessEqual(two["economy"], 100.0)
        self.assertGreaterEqual(two["lci_score"], 0.0)
        self.assertLessEqual(two["lci_score"], 100.0)
        self.assertAlmostEqual(two["actual_cost"], two["lifecycle_cost"])

    def test_25_windows_scripts_do_not_depend_on_py_only(self):
        batch = (ROOT / "start_windows.bat").read_text(encoding="utf-8")
        powershell = (ROOT / "start_windows.ps1").read_text(encoding="utf-8")
        self.assertIn(".venv\\Scripts\\python.exe", batch)
        self.assertIn("python --version", batch)
        self.assertIn("py -3 --version", batch)
        self.assertIn("Get-Command python", powershell)
        self.assertIn("Get-Command py", powershell)

    def test_26_frontend_has_no_prohibited_positioning_terms(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        prohibited = ["最小可运行原型", "MVP", "概念验证", "简易演示", "Demo版本", "AI生成", "D类参数", "NSGA-II尚未实现", "仅供演示"]
        self.assertFalse([term for term in prohibited if term in app])

    def test_27_json_export_is_chinese_and_parseable(self):
        payload = json.loads(result_to_json(self.exact))
        self.assertIn("运行元数据", payload)
        self.assertIn("字段中英文映射", payload)
        self.assertIn("Pareto候选方案", payload)
        self.assertEqual(len(payload["Pareto候选方案"]), len(self.exact.pareto))


if __name__ == "__main__":
    unittest.main()
