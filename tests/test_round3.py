from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.benchmarks import evaluate_benchmark, quality_metrics, theoretical_front
from model.constraints import check_plan_constraints
from model.data_validation import validate_project_data
from model.evaluation import evaluate_unit
from model.i18n import chinese_frame, unmapped_columns
from model.nsga2 import solve_nsga2
from model.schemas import SolverConfig, load_measures, load_units
from model.screening import screen_all
from model.solver import solve_exact
from model.spatial import coordination_metrics, load_geojson, load_topology
from model.validation import validate_runs
from model.workflow import extract_unit_id, next_step, optimization_hash, preference_hash, step_validation, synchronize_selected_unit


class RoundThreeAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units_df = pd.read_csv(ROOT / "data/default_units.csv")
        cls.measures_df = pd.read_csv(ROOT / "data/default_measures.csv")
        cls.units = load_units(ROOT / "data/default_units.csv")
        cls.measures = load_measures(ROOT / "data/default_measures.csv")
        cls.geojson = load_geojson(ROOT / "data/spatial_units.geojson")
        cls.topology = load_topology(ROOT / "data/spatial_topology.json")
        cls.config = SolverConfig(nsga_population_size=40, nsga_generations=5, random_seed=42)
        cls.exact = solve_exact(cls.units, cls.measures, cls.config, ROOT / "data/assumptions.csv")
        cls.nsga = solve_nsga2(cls.units, cls.measures, cls.config, ROOT / "data/assumptions.csv")

    def test_28_next_step_advances_without_reset(self):
        self.assertEqual(next_step(2, True), 3)

    def test_29_invalid_step_cannot_advance(self):
        self.assertEqual(next_step(2, False), 2)

    def test_30_current_step_survives_equivalent_rerun(self):
        current = 4
        self.assertEqual(next_step(current, False), current)

    def test_31_map_click_returns_unit_id(self):
        event = {"selection": {"points": [{"customdata": ["U03", "沟谷单元"]}]}}
        self.assertEqual(extract_unit_id(event, ["U01", "U02", "U03", "U04"]), "U03")

    def test_32_map_click_changes_detail_target(self):
        event = {"selection": {"points": [{"location": "U02"}]}}
        target = extract_unit_id(event, self.units_df.unit_id.tolist())
        row = self.units_df[self.units_df.unit_id == target].iloc[0]
        self.assertEqual(row.unit_name, "低风险浅表冲刷缓坡")

    def test_33_table_selection_synchronizes_map(self):
        self.assertEqual(synchronize_selected_unit("U04", self.units_df.unit_id.tolist()), "U04")

    def test_34_plan_changes_path_metrics(self):
        untreated = [{"unit_id": uid, "measures": "不治理"} for uid in self.units_df.unit_id]
        controlled = [{"unit_id": "U01", "measures": "挡墙工程"}, {"unit_id": "U02", "measures": "柔性拦挡结构"},
                      {"unit_id": "U03", "measures": "格构锚固＋谷坊坝"}, {"unit_id": "U04", "measures": "截排水工程＋柔性拦挡结构"}]
        self.assertGreater(coordination_metrics(self.topology, controlled)["上下游协同贡献（%）"], coordination_metrics(self.topology, untreated)["上下游协同贡献（%）"])

    def test_35_detail_text_not_ellipsized(self):
        source = (ROOT / "ui_v3.py").read_text(encoding="utf-8")
        self.assertNotIn("text-overflow:ellipsis", source.replace(" ", ""))

    def test_36_online_failure_has_offline_fallback(self):
        source = (ROOT / "ui_v3.py").read_text(encoding="utf-8")
        self.assertIn('map_figure(plan_details, st.session_state.selected_unit, False)', source)

    def test_37_csv_geojson_topology_ids_match(self):
        self.assertEqual(validate_project_data(self.units_df, self.measures_df, self.geojson, self.topology), [])

    def test_38_deleted_unit_is_detected(self):
        errors = validate_project_data(self.units_df.iloc[:-1], self.measures_df, self.geojson, self.topology)
        self.assertTrue(any("不一致" in item or "不存在" in item for item in errors))

    def test_39_frontend_has_no_code_or_markdown_widget(self):
        source = (ROOT / "ui_v3.py").read_text(encoding="utf-8")
        self.assertNotIn("st.code(", source)
        self.assertNotIn("```python", source)

    def test_40_screening_ids_are_unique(self):
        frame = pd.DataFrame([item.to_dict() for item in screen_all(self.units, self.measures)])
        self.assertEqual(len(frame), 32)
        self.assertEqual(frame.screening_id.nunique(), 32)
        self.assertRegex(frame.screening_id.iloc[0], r"^U\d{2}-M\d{2}$")

    def test_41_frontend_frames_have_chinese_headers(self):
        frame = pd.DataFrame([item.to_dict() for item in screen_all(self.units, self.measures)])
        self.assertEqual(unmapped_columns(frame), [])
        self.assertFalse(any(re.fullmatch(r"[a-z][a-z0-9_]*", str(x)) for x in chinese_frame(frame).columns))

    def test_42_exact_enumeration_reproducible(self):
        repeated = solve_exact(self.units, self.measures, self.config, ROOT / "data/assumptions.csv")
        pd.testing.assert_frame_equal(self.exact.pareto, repeated.pareto)

    def test_43_nsga_same_seed_reproducible(self):
        repeated = solve_nsga2(self.units, self.measures, self.config, ROOT / "data/assumptions.csv")
        pd.testing.assert_frame_equal(self.nsga.pareto, repeated.pareto)

    def test_44_nsga_outputs_satisfy_hard_constraints(self):
        for details in self.nsga.detail_lookup.values():
            valid, reasons = check_plan_constraints(self.units, details, sum(float(x["lifecycle_cost"]) for x in details), self.config)
            self.assertTrue(valid, reasons)

    def test_45_nsga_has_no_unadmitted_measure(self):
        screening = pd.DataFrame([item.to_dict() for item in screen_all(self.units, self.measures)])
        eligible = {(r.unit_id, r.measure_name) for r in screening[screening.eligible].itertuples()}
        for details in self.nsga.detail_lookup.values():
            for item in details:
                for name in str(item["measures"]).split("＋"):
                    if name != "不治理": self.assertIn((item["unit_id"], name), eligible)

    def test_46_nsga_and_exact_share_evaluation(self):
        exact = self.exact.all_feasible.set_index("configuration_summary")
        common = self.nsga.all_feasible[self.nsga.all_feasible.configuration_summary.isin(exact.index)]
        self.assertGreater(len(common), 0)
        row = common.iloc[0]; expected = exact.loc[row.configuration_summary]
        self.assertAlmostEqual(row.safety, expected.safety, places=6)

    def test_47_standard_function_metrics_correct(self):
        truth = theoretical_front("ZDT1", 100)
        metrics = quality_metrics(truth, truth)
        self.assertLess(metrics["IGD"], 1e-12)
        self.assertLess(metrics["GD"], 1e-12)
        values = evaluate_benchmark("DTLZ2", np.full((1, 12), .5))[0]
        self.assertAlmostEqual(float(np.linalg.norm(values)), 1.0, places=6)

    def test_48_multiseed_validation_records_real_seeds(self):
        runs = []
        for seed in (42, 123):
            runs.append(solve_nsga2(self.units, self.measures, replace(self.config, random_seed=seed), ROOT / "data/assumptions.csv"))
        summary = validate_runs(self.exact, runs).seed_summary
        self.assertEqual(set(summary["随机种子"]), {42, 123})

    def test_49_economy_monotonic_with_added_cost(self):
        drainage = next(x for x in self.measures if x.measure_name == "截排水工程")
        grid = next(x for x in self.measures if x.measure_name == "格构锚固")
        one = evaluate_unit(self.units[0], (grid,), self.config); two = evaluate_unit(self.units[0], (grid, drainage), self.config)
        self.assertGreater(two["lifecycle_cost"], one["lifecycle_cost"]); self.assertLess(two["economy"], one["economy"])

    def test_50_untreated_scores_are_not_artificially_high(self):
        result = evaluate_unit(self.units[0], (), self.config)
        self.assertLessEqual(result["sustainability"], 50)
        self.assertLessEqual(result["resilience"], 35)

    def test_51_path_requires_stage_matching_measure(self):
        ecological = [{"unit_id": uid, "measures": "灌草种植"} for uid in self.units_df.unit_id]
        metrics = coordination_metrics(self.topology, ecological)
        self.assertEqual(metrics["完整覆盖路径数"], 0)
        self.assertEqual(metrics["输移拦截率（%）"], 0)

    def test_52_input_change_invalidates_optimization_hash(self):
        first = optimization_hash(self.units, self.measures, self.config, "exact")
        second = optimization_hash(self.units, self.measures, replace(self.config, total_budget=400), "exact")
        self.assertNotEqual(first, second)

    def test_53_preference_change_does_not_require_rerun(self):
        first = optimization_hash(self.units, self.measures, self.config, "exact")
        changed = replace(self.config, preference_weights={"safety": 1, "economy": 0, "sustainability": 0, "resilience": 0})
        self.assertEqual(first, optimization_hash(self.units, self.measures, changed, "exact"))
        self.assertNotEqual(preference_hash(self.config.preference_weights), preference_hash(changed.preference_weights))

    def test_54_exports_are_defined_for_csv_json_html_zip(self):
        source = (ROOT / "ui_v3.py").read_text(encoding="utf-8")
        for token in ("to_csv", "result_to_json", "build_html_report", "ZipFile"):
            self.assertIn(token, source)

    def test_55_windows_launcher_detection_order(self):
        batch = (ROOT / "start_windows.bat").read_text(encoding="utf-8")
        self.assertLess(batch.index(".venv\\Scripts\\python.exe"), batch.index("python --version"))
        self.assertLess(batch.index("python --version"), batch.index("py -3 --version"))

    def test_56_requirements_cover_runtime_and_tests(self):
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for package in ("numpy", "pandas", "plotly", "streamlit", "pytest"):
            self.assertIn(package, req)

    def test_57_layout_has_1920_workbench_and_responsive_rule(self):
        source = (ROOT / "ui_v3.py").read_text(encoding="utf-8")
        self.assertIn("max-width:1760px", source)
        self.assertIn("@media(max-width:1200px)", source)


if __name__ == "__main__":
    unittest.main()
