"""现有实现中尚待标定的参数集中管理；默认值保持V4基线。"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EvaluationParameters:
    status: str = "V1暂定参数，待工程标定"
    fs_score_zero: float = 0.80
    fs_score_full: float = 1.40
    chain_base: float = 20.0
    delta_fs_cap: float = 0.70
    steep_slope_deg: float = 45.0
    tall_slope_m: float = 20.0
    steep_cost_addition: float = 0.08
    tall_cost_addition: float = 0.07
    inaccessible_cost_addition: float = 0.12
    steep_disturbance_penalty: float = 5.0
    untreated_sustainability: float = 45.0
    untreated_redundancy: float = 15.0
    untreated_recovery_difficulty: float = 3.0
    maintenance_base: float = 0.03
    maintenance_level_factor: float = 0.015
    organization_level_factor: float = 0.02
    legacy_cost_score_factor: float = 0.35
    residual_hazard_fs_factor: float = 4.0
    residual_hazard_chain_divisor: float = 70.0
    residual_risk_fs_factor: float = 3.0
    residual_risk_chain_divisor: float = 80.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == "status":
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value < 0:
                raise ValueError(f"评价参数{name}须为有限非负数。")
        if self.fs_score_full <= self.fs_score_zero:
            raise ValueError("稳定性映射满分点须大于零分点。")
        if min(self.residual_hazard_chain_divisor, self.residual_risk_chain_divisor) <= 0:
            raise ValueError("残余风险映射分母须大于零。")
        if self.untreated_sustainability > 100 or self.untreated_redundancy > 100:
            raise ValueError("未治理基线得分须在0～100之间。")
        if not 1 <= self.untreated_recovery_difficulty <= 5:
            raise ValueError("未治理恢复难度须在1～5之间。")
