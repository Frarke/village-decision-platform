from __future__ import annotations

from .schemas import GovernanceUnit, Measure, ScreeningResult
from .tagging import identify_tags


def screen_measure(unit: GovernanceUnit, measure: Measure) -> ScreeningResult:
    tags, _ = identify_tags(unit)
    tag_set = set(tags)
    name = measure.measure_name
    entry: list[str] = []
    failures: list[str] = []
    combo: list[str] = []
    review: list[str] = []

    if name == "截排水工程":
        needed = "降雨敏感" in tag_set or unit.catchment_km2 >= 0.02 or unit.risk >= 4
        if needed:
            entry.append("降雨敏感、坡面汇流或中高风险排水需求")
        else:
            failures.append("未识别明显降雨/汇流/排水需求")
        if not unit.safe_drainage_outlet:
            failures.append("无安全排水出路且下游无法保证消能导排")
    elif name == "灌草种植":
        if unit.slope_deg <= 25 and unit.risk <= 3:
            entry.append("θ≤25°且Risk≤3，可作为单项候选")
        elif 25 < unit.slope_deg <= 35:
            entry.append("25°<θ≤35°，仅作为组合候选")
            combo.append("必须与截排水工程组合")
        elif unit.risk >= 4 and unit.slope_deg <= 35:
            entry.append("生态恢复辅助需求")
        else:
            failures.append("坡度超出灌草种植候选范围")
        if unit.risk >= 4:
            combo.append("Risk≥4，不得作为单独主控措施，须配置主控工程")
    elif name == "植被混凝土":
        if 25 <= unit.slope_deg <= 55 and unit.height_m <= 20 and unit.artificial_or_bare:
            entry.append("25°≤θ≤55°、H≤20m且为人工切坡/裸露坡面")
        else:
            failures.append("坡度、坡高或坡面类型不满足进入条件")
        if unit.risk >= 4:
            combo.append("Risk≥4，不得单独作为主控措施，须配置主控工程")
    elif name == "格构锚固":
        if 30 <= unit.slope_deg <= 60 and 5 <= unit.height_m <= 30:
            entry.append("30°≤θ≤60°且5m≤H≤30m")
        else:
            failures.append("坡度或坡高不在格构锚固适用范围")
        if not unit.anchor_condition:
            failures.append("无有效锚固条件")
        if not unit.accessible:
            failures.append("施工不可达")
    elif name == "挡墙工程":
        if unit.height_m <= 10 and unit.foot_space_m >= 2:
            entry.append("H≤10m且W≥2m")
        else:
            failures.append("H>10m或坡脚施工空间W<2m")
        if not unit.wall_back_drainage:
            failures.append("墙后排水无法保证")
    elif name == "抗滑桩":
        if unit.risk >= 4 or unit.haz >= 4:
            entry.append("Risk≥4或Haz≥4")
        else:
            failures.append("未达到中高风险滑移准入等级")
        if not unit.slip_surface_or_review:
            failures.append("无滑面资料且不具备后续勘察复核条件")
        elif not unit.slip_surface_confirmed:
            review.append("需勘察复核滑面及稳定嵌固段，当前不得视为最终工程结论")
    elif name == "谷坊坝":
        is_gully = unit.gully_level >= 2
        if is_gully and unit.catchment_km2 >= 0.05:
            entry.append("沟谷单元且Ac≥0.05km²")
        else:
            failures.append("非沟谷单元或Ac<0.05km²")
    elif name == "柔性拦挡结构":
        impact_zone = unit.motion_path and (unit.gully_level >= 1 or unit.distance_m <= 100)
        if impact_zone and unit.anchor_condition:
            entry.append("存在明确运动路径/冲击区且具备锚固条件")
        else:
            failures.append("无明确运动路径、冲击区或有效锚固条件")
        if unit.risk >= 4:
            combo.append("高风险单元不得作为单独主控措施")
    else:
        failures.append("未知措施类型")

    eligible = not failures
    return ScreeningResult(
        unit_id=unit.unit_id,
        unit_name=unit.unit_name,
        measure_id=measure.measure_id,
        measure_name=name,
        eligible=eligible,
        entry_conditions="；".join(entry) if entry else "未命中",
        failed_restrictions="；".join(failures) if failures else "无",
        exclusion_reason="；".join(failures) if failures else "—",
        combination_required="；".join(dict.fromkeys(combo)) if combo else "无",
        review_required="；".join(review) if review else "无",
    )


def screen_all(units: list[GovernanceUnit], measures: list[Measure]) -> list[ScreeningResult]:
    return [screen_measure(unit, measure) for unit in units for measure in measures]
