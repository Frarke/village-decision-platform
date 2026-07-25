from __future__ import annotations

from itertools import combinations
from math import prod

from .schemas import GovernanceUnit, Measure, ScreeningResult, SolverConfig


MUTUALLY_EXCLUSIVE = [
    frozenset({"灌草种植", "植被混凝土"}),
    frozenset({"挡墙工程", "抗滑桩"}),
]


def validate_unit_combination(
    unit: GovernanceUnit,
    selected: tuple[Measure, ...],
    config: SolverConfig,
) -> tuple[bool, str]:
    names = {measure.measure_name for measure in selected}
    if not selected:
        if unit.risk >= 3 and not config.allow_untreated_medium_high:
            return False, "配置禁止中高风险单元不治理"
        return True, "允许不治理"
    if len(selected) > config.max_measures_per_unit:
        return False, "超过每单元最大措施数量"
    for pair in MUTUALLY_EXCLUSIVE:
        if pair.issubset(names):
            return False, f"互斥措施：{'＋'.join(sorted(pair))}"
    if "灌草种植" in names and 25 < unit.slope_deg <= 35 and "截排水工程" not in names:
        return False, "25°<θ≤35°时灌草种植必须与截排水工程组合"
    primary = any(measure.is_primary_control for measure in selected)
    if unit.risk >= 4:
        if names.intersection({"灌草种植", "植被混凝土"}) and not primary:
            return False, "高风险单元的生态措施不能替代主控工程"
        if names == {"柔性拦挡结构"}:
            return False, "高风险单元不能仅采用柔性拦挡结构"
    return True, "通过组合约束"


def generate_unit_options(
    unit: GovernanceUnit,
    measures: list[Measure],
    screening: list[ScreeningResult],
    config: SolverConfig,
) -> list[tuple[Measure, ...]]:
    eligible_ids = {
        result.measure_id
        for result in screening
        if result.unit_id == unit.unit_id and result.eligible
    }
    candidates = [measure for measure in measures if measure.measure_id in eligible_ids]
    options: list[tuple[Measure, ...]] = []
    sizes = range(0, min(config.max_measures_per_unit, len(candidates)) + 1)
    for size in sizes:
        for combo in combinations(candidates, size):
            valid, _ = validate_unit_combination(unit, combo, config)
            if valid:
                options.append(combo)
    options.sort(key=lambda combo: (len(combo), tuple(m.measure_id for m in combo)))
    return options


def estimate_enumerations(options_by_unit: dict[str, list[tuple[Measure, ...]]]) -> int:
    return prod(len(options) for options in options_by_unit.values()) if options_by_unit else 0
