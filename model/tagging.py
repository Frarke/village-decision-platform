from __future__ import annotations

from .schemas import GovernanceUnit


TAG_RULES = {
    "降雨敏感": "P≥1600 mm",
    "浅表坡面冲刷": "θ≤25°、H≤10 m且Ac<0.02 km²",
    "人工切坡或裸露坡面": "H≥5 m、θ≥25°且识别为人工切坡/裸露坡面",
    "中高风险滑移": "Risk≥4或Haz≥4",
    "沟谷汇流或输移": "Ac≥0.05 km²或G≥2",
    "近承灾体": "L≤100 m",
    "坡脚施工受限": "W<2 m",
}


def in_governance_library(unit: GovernanceUnit) -> tuple[bool, str]:
    reasons: list[str] = []
    if unit.risk >= 3:
        reasons.append(f"Risk={unit.risk}≥3")
    if unit.distance_m <= 100:
        reasons.append(f"L={unit.distance_m:g}m≤100m")
    if unit.identified_or_risk_area:
        reasons.append("已识别隐患点/风险区")
    if unit.focus_unit:
        reasons.append("重点关注单元")
    return bool(reasons), "；".join(reasons) if reasons else "未命中入库条件"


def identify_tags(unit: GovernanceUnit) -> tuple[list[str], dict[str, str]]:
    tags: list[str] = []
    evidence: dict[str, str] = {}

    def add(condition: bool, tag: str, reason: str) -> None:
        if condition:
            tags.append(tag)
            evidence[tag] = reason

    add(unit.rainfall_mm >= 1600, "降雨敏感", f"P={unit.rainfall_mm:g}mm≥1600mm")
    add(
        unit.slope_deg <= 25 and unit.height_m <= 10 and unit.catchment_km2 < 0.02,
        "浅表坡面冲刷",
        f"θ={unit.slope_deg:g}°≤25°、H={unit.height_m:g}m≤10m、Ac={unit.catchment_km2:g}<0.02km²",
    )
    add(
        unit.height_m >= 5 and unit.slope_deg >= 25 and unit.artificial_or_bare,
        "人工切坡或裸露坡面",
        f"H={unit.height_m:g}m≥5m、θ={unit.slope_deg:g}°≥25°且人工切坡/裸露",
    )
    add(unit.risk >= 4 or unit.haz >= 4, "中高风险滑移", f"Risk={unit.risk}，Haz={unit.haz}")
    add(
        unit.catchment_km2 >= 0.05 or unit.gully_level >= 2,
        "沟谷汇流或输移",
        f"Ac={unit.catchment_km2:g}km²，G={unit.gully_level}",
    )
    add(unit.distance_m <= 100, "近承灾体", f"L={unit.distance_m:g}m≤100m")
    add(unit.foot_space_m < 2, "坡脚施工受限", f"W={unit.foot_space_m:g}m<2m")
    return tags, evidence


def tagged_unit_record(unit: GovernanceUnit) -> dict[str, object]:
    included, inclusion_reason = in_governance_library(unit)
    tags, evidence = identify_tags(unit)
    record = unit.to_dict()
    record.update({
        "in_governance_library": included,
        "inclusion_reason": inclusion_reason,
        "auto_tags": "；".join(tags) if tags else "无",
        "tag_evidence": " | ".join(f"{tag}：{reason}" for tag, reason in evidence.items()) if evidence else "无",
    })
    return record
