from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.combinations import validate_unit_combination
from model.constraints import check_plan_constraints
from model.evaluation import evaluate_unit
from model.pareto import dominates, pareto_front
from model.regional import evaluate_region
from model.schemas import SolverConfig, load_measures, load_units
from model.screening import screen_measure
from model.solver import solve_exact
from model.tagging import identify_tags, in_governance_library


class ModelAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = load_units(ROOT / "data/default_units.csv")
        cls.measures = load_measures(ROOT / "data/default_measures.csv")
        cls.config = SolverConfig()

    @classmethod
    def by_name(cls, name):
        return next(measure for measure in cls.measures if measure.measure_name == name)

    def test_01_unit_inclusion_and_tags(self):
        included, reason = in_governance_library(self.units[0])
        tags, evidence = identify_tags(self.units[0])
        self.assertTrue(included)
        self.assertIn("Risk=4≥3", reason)
        self.assertTrue({"降雨敏感", "人工切坡或裸露坡面", "中高风险滑移", "近承灾体", "坡脚施工受限"}.issubset(tags))
        self.assertTrue(all(tag in evidence for tag in tags))

    def test_02_wall_rejected_when_foot_space_too_small(self):
        result = screen_measure(self.units[0], self.by_name("挡墙工程"))
        self.assertFalse(result.eligible)
        self.assertIn("W<2m", result.exclusion_reason)

    def test_03_check_dam_rejected_for_non_gully(self):
        result = screen_measure(self.units[1], self.by_name("谷坊坝"))
        self.assertFalse(result.eligible)
        self.assertIn("非沟谷单元", result.exclusion_reason)

    def test_04_anchor_measures_rejected_without_anchor(self):
        no_anchor = replace(self.units[0], anchor_condition=False)
        grid = screen_measure(no_anchor, self.by_name("格构锚固"))
        barrier = screen_measure(no_anchor, self.by_name("柔性拦挡结构"))
        self.assertFalse(grid.eligible)
        self.assertIn("无有效锚固条件", grid.exclusion_reason)
        self.assertFalse(barrier.eligible)
        self.assertIn("锚固", barrier.exclusion_reason)

    def test_05_ecological_measure_not_high_risk_primary(self):
        high_risk = self.units[0]
        self.assertFalse(validate_unit_combination(high_risk, (self.by_name("灌草种植"),), self.config)[0])
        self.assertFalse(validate_unit_combination(high_risk, (self.by_name("植被混凝土"),), self.config)[0])

    def test_06_over_budget_plan_rejected(self):
        details = [evaluate_unit(unit, (self.by_name("抗滑桩"),) if i == 0 else (), self.config) for i, unit in enumerate(self.units)]
        valid, reasons = check_plan_constraints(self.units, details, 9999, self.config)
        self.assertFalse(valid)
        self.assertTrue(any("预算" in reason for reason in reasons))

    def test_07_safety_floor_rejects_unsafe_plan(self):
        details = [evaluate_unit(unit, (), self.config) for unit in self.units]
        valid, reasons = check_plan_constraints(self.units, details, 0, self.config)
        self.assertFalse(valid)
        self.assertTrue(any("FS" in reason or "安全得分" in reason for reason in reasons))

    def test_08_pareto_dominance(self):
        self.assertTrue(dominates(np.array([80, 70, 60, 50]), np.array([70, 70, 50, 50])))
        self.assertFalse(dominates(np.array([80, 40, 60, 50]), np.array([70, 70, 50, 50])))
        frame = pd.DataFrame([
            {"plan_id": "A", "total_cost": 1, "safety": 80, "economy": 70, "sustainability": 60, "resilience": 50},
            {"plan_id": "B", "total_cost": 2, "safety": 70, "economy": 70, "sustainability": 50, "resilience": 50},
            {"plan_id": "C", "total_cost": 3, "safety": 60, "economy": 90, "sustainability": 80, "resilience": 70},
        ])
        self.assertEqual(set(pareto_front(frame).plan_id), {"A", "C"})

    def test_09_regional_safety_formula(self):
        subset = self.units[:2]
        details = [
            {"unit_id": subset[0].unit_id, "safety": 50, "economy": 60, "sustainability": 70, "resilience": 80, "actual_cost": 10},
            {"unit_id": subset[1].unit_id, "safety": 90, "economy": 80, "sustainability": 90, "resilience": 60, "actual_cost": 20},
        ]
        result = evaluate_region(subset, details)
        expected = 0.6 * 50 + 0.4 * ((4.5 / 6.0) * 50 + (1.5 / 6.0) * 90)
        self.assertAlmostEqual(result["safety"], expected)

    def test_10_zero_key_paths_no_division_error(self):
        result = evaluate_unit(self.units[1], (), self.config)
        self.assertEqual(result["key_paths"], 0)
        self.assertTrue(np.isfinite(result["resilience"]))

    def test_11_default_case_nonempty(self):
        result = solve_exact(self.units, self.measures, self.config, ROOT / "data/assumptions.csv")
        self.assertGreater(len(result.all_feasible), 0)
        self.assertGreater(len(result.pareto), 1)
        self.assertEqual(len(result.representatives), 5)

    def test_12_budget_change_affects_results(self):
        high = solve_exact(self.units, self.measures, SolverConfig(total_budget=520), ROOT / "data/assumptions.csv")
        low = solve_exact(self.units, self.measures, SolverConfig(total_budget=350), ROOT / "data/assumptions.csv")
        self.assertLess(len(low.all_feasible), len(high.all_feasible))
        self.assertNotEqual(high.representatives.configuration_summary.tolist(), low.representatives.configuration_summary.tolist())

    def test_13_outputs_have_no_nan_or_infinity(self):
        result = solve_exact(self.units, self.measures, self.config, ROOT / "data/assumptions.csv")
        frames = [result.tagged_units, result.screening, result.all_feasible, result.pareto, result.representatives, result.representative_details]
        for frame in frames:
            numeric = frame.select_dtypes(include="number")
            self.assertTrue(numeric.notna().all().all())
            self.assertTrue(np.isfinite(numeric.to_numpy()).all())

    def test_14_fixed_input_is_deterministic(self):
        first = solve_exact(self.units, self.measures, self.config, ROOT / "data/assumptions.csv")
        second = solve_exact(self.units, self.measures, self.config, ROOT / "data/assumptions.csv")
        pd.testing.assert_frame_equal(first.all_feasible, second.all_feasible)
        pd.testing.assert_frame_equal(first.pareto, second.pareto)
        pd.testing.assert_frame_equal(first.representatives, second.representatives)

    def test_15_antislide_review_flag(self):
        result = evaluate_unit(self.units[0], (self.by_name("抗滑桩"),), self.config)
        self.assertIn("勘察复核", result["expert_review"])


if __name__ == "__main__":
    unittest.main()
