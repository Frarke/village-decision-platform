"""集中维护前台与导出字段中文名称，避免各页面散落硬编码。"""

from __future__ import annotations

import pandas as pd


FIELD_LABELS: dict[str, str] = {
    "unit_id": "治理单元编号", "unit_name": "治理单元名称", "risk": "风险等级",
    "haz": "危险性等级", "distance_m": "与承灾体距离（米）", "rainfall_mm": "评价降雨量（毫米）",
    "slope_deg": "地形坡度（度）", "height_m": "坡高（米）", "catchment_km2": "汇流面积（平方千米）",
    "gully_level": "沟谷位置等级", "foot_space_m": "坡脚施工空间（米）", "fs0": "现状稳定系数",
    "k_rain": "降雨折减系数", "identified_or_risk_area": "已识别隐患点或风险区",
    "focus_unit": "重点关注单元", "safe_drainage_outlet": "具备安全排水出路",
    "artificial_or_bare": "人工切坡或裸露坡面", "anchor_condition": "具备有效锚固条件",
    "slip_surface_or_review": "具备滑面资料或勘察复核条件", "slip_surface_confirmed": "滑面已确认",
    "motion_path": "存在明确运动路径", "accessible": "道路或施工可达",
    "wall_back_drainage": "墙后排水可保证", "key_paths": "关键风险路径数量",
    "risk_weight": "风险权重", "evidence_level": "参数依据状态", "notes": "参数备注",
    "in_governance_library": "是否进入治理对象库", "inclusion_reason": "入库判定依据",
    "auto_tags": "自动识别标签", "tag_evidence": "标签判定依据",
    "measure_id": "措施编号", "measure_name": "工程措施名称", "function_type": "功能类型",
    "screening_id": "准入判定唯一编号",
    "applicable_objects": "主要适用对象", "actual_cost": "基础建设成本（万元）",
    "build_level": "建设成本等级", "maintain_level": "维护成本等级", "construct_level": "施工组织难度等级",
    "delta_fs": "稳定性提升值", "chain_delta": "连锁防控能力增量", "eco_score": "生态恢复得分",
    "carbon_level": "碳排放等级", "disturbance_level": "工程扰动等级",
    "recovery_difficulty": "恢复维护难度等级", "covered_paths": "可覆盖关键路径数量",
    "is_primary_control": "是否为主控工程", "eligible": "是否准入", "entry_conditions": "命中的进入条件",
    "failed_restrictions": "未通过的限制条件", "exclusion_reason": "剔除原因",
    "combination_required": "组合要求", "review_required": "专家或勘察复核要求",
    "plan_id": "方案编号", "total_cost": "全寿命周期总成本（万元）", "safety": "区域安全性得分",
    "economy": "区域经济性得分", "sustainability": "区域可持续性得分", "resilience": "区域韧性得分",
    "configuration_summary": "村域方案配置", "expert_review": "专家复核事项",
    "recommendation_type": "代表方案类型", "recommendation_basis": "推荐依据",
    "measures": "当前措施配置", "measure_ids": "措施内部编号", "construction_cost": "建设成本（万元）",
    "maintenance_npv": "维护折现成本（万元）", "organization_cost": "施工组织成本（万元）",
    "terrain_adjustment_cost": "地形修正成本（万元）", "lifecycle_cost": "全寿命周期成本（万元）",
    "cost_factor": "地形成本修正系数", "fs_rain": "治理后降雨工况稳定系数",
    "fs_before_rain": "治理前降雨工况稳定系数",
    "synergy_factor": "组合协同系数", "s_fs": "稳定性分项得分", "s_chain": "连锁防控分项得分",
    "lci": "相对生命周期成本指数", "lci_score": "报告口径经济性基础得分",
    "lifecycle_cost_score": "全寿命周期成本得分",
    "measure_names": "措施名称列表",
    "redundancy_score": "冗余防线得分",
    "recovery_score": "恢复能力得分",
    "coverage_score_local": "单元路径覆盖得分",
    "path_coverage": "关键路径覆盖率（%）",
    "coordination_score": "区域协同得分（%）",
    "N_key": "关键路径总数",
    "N_covered": "已覆盖关键路径数",
    "normalized_risk_weights": "归一化风险权重",
    "constraint_status": "约束满足状态",
    "constraint_reasons": "约束违反原因",
    "tie_candidate_count": "主目标并列候选数",
    "tie_candidate_ids": "主目标并列候选编号",
    "tie_break_rule": "并列破除规则",
    "primary_indicator": "主判定指标",
    "primary_value": "主判定指标值",
    "primary_ascending": "主判定升序",
    "preference_weights": "偏好权重",
    "compromise_distance": "综合折中距离",
    "weakest_objective": "最弱目标值",
    "function_count": "有效功能防线数量", "key_paths": "关键风险路径总数",
    "residual_hazard": "治理后残余危险性等级", "residual_risk": "治理后残余风险等级",
    "risk_reduction": "风险降低幅度", "downstream_transfer_review": "下游风险转移复核",
    "scenario_id": "场景编号", "configuration_hash": "配置唯一标识", "solver": "求解方法",
    "generation": "迭代代数", "current_feasible": "当前可行个体数", "current_nondominated": "当前非支配个体数",
    "unique_evaluations": "累计唯一方案评价数", "convergence_distance": "历史兼容指标",
    "archive_hypervolume": "累计档案归一化超体积", "front_spacing": "非支配前沿间距离散度",
    "search_space_ratio": "实际搜索空间占比",
    "total_budget": "总预算（万元）", "max_measures_per_unit": "每单元最多措施数量",
    "allow_untreated_medium_high": "允许中高风险单元不治理", "high_risk_min_fs": "高风险稳定系数底线",
    "high_risk_min_safety": "高风险安全得分底线", "max_enumerations": "最大精确枚举数",
    "drainage_synergy": "排水组合协同系数", "general_second_measure_factor": "第二项措施效果折减系数",
    "nsga_population_size": "NSGA-II种群规模", "nsga_generations": "NSGA-II最大迭代代数",
    "nsga_crossover_probability": "NSGA-II交叉概率", "nsga_mutation_probability": "NSGA-II变异概率",
    "random_seed": "随机种子", "preference_weights": "代表方案偏好权重",
    "safety": "安全性", "economy": "经济性", "sustainability": "可持续性", "resilience": "韧性",
}

VALUE_LABELS = {
    True: "是", False: "否", "D": "待标定参数", "exact_enumeration": "精确枚举",
    "nsga2_discrete": "NSGA-II离散智能搜索",
}


def chinese_frame(frame: pd.DataFrame, include_internal: bool = False) -> pd.DataFrame:
    """返回中文字段副本；内部字段可按需保留以支持追溯。"""
    result = frame.copy()
    if not include_internal:
        result = result.drop(columns=[c for c in result.columns if c.startswith("unit_") or c == "measure_ids"], errors="ignore")
    for column in result.columns:
        if result[column].dtype == bool:
            result[column] = result[column].map({True: "是", False: "否"})
        elif column == "evidence_level":
            result[column] = result[column].replace({"D": "待标定参数"})
    return result.rename(columns={name: FIELD_LABELS.get(name, name) for name in result.columns})


def unmapped_columns(frame: pd.DataFrame, allow_prefixes: tuple[str, ...] = ("unit_",)) -> list[str]:
    return [name for name in frame.columns if name not in FIELD_LABELS and not name.startswith(allow_prefixes)]
