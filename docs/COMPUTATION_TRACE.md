# 计算链追踪表

本文档用于对照 `references_private/tech_report/中期科技报告.docx` 与当前 `model/` 实现，记录正式公式、代码入口、默认参数与待标定边界。

## 1. 候选方案统一评价链

当前 Python 主评价入口：

- `model/regional.py:evaluate_village_scheme`
- `model/solver.py:solve_exact`
- `model/nsga2.py:solve_nsga2`

当前执行顺序：

1. `screen_all()` 先形成措施准入结果。
2. `generate_unit_options()` 基于准入结果生成各治理单元候选措施组合。
3. `evaluate_village_scheme()` 对每个候选村域方案校验编码、准入、组合和拓扑，再执行统一评价；非法候选不会在入口中自动修复。
4. `evaluate_unit()` 计算单元四目标与成本分解。
5. `evaluate_region()` 执行风险权重归一化与区域聚合。
6. `evaluate_path_coverage()` 计算关键路径覆盖、区域协同与韧性相关路径指标。
7. 单元局部路径覆盖与韧性从同一批 `path_details` 同步；`check_plan_constraints()` 对预算、安全底线和未治理约束做统一判定。
8. 精确枚举与 NSGA-II 仅改变候选方案搜索方式，不改变评价链。

## 2. 科技报告公式与代码实现

| 科技报告公式/规则 | 当前代码位置 | 主要输入 | 默认/当前实现 | 一致性判断 |
| --- | --- | --- | --- | --- |
| `FSrain = FS0 × Krain + ΔFSp` | `model/evaluation.py:evaluate_unit` | `unit.fs0`, `unit.k_rain`, `measure.delta_fs` | 已实现；组合措施用 `_combined_delta_fs()` 聚合，并带排水协同系数 | 基本一致；组合叠加细则属平台当前规则，仍需标定 |
| `S = 0.6 SFS + 0.4 S_chain` | `model/evaluation.py:evaluate_unit` | `fs_rain`, `measure.chain_delta` | 已实现；`SFS` 由 FS 线性映射得到 | 结构一致；`SFS` 映射区间为当前规则，待标定 |
| `LCI = 0.65 Cbuild + 0.20 Cmaint + 0.15 Cconstruct` | `model/evaluation.py:evaluate_unit` | `measure.build_level`, `maintain_level`, `construct_level` | 已实现，按措施均值求单元 `lci` | 一致 |
| `SC = 100 − 20 × (LCI − 1)` | `model/evaluation.py:evaluate_unit` | `lci` | 已实现为 `lci_score`，并用于 `economy` | 一致 |
| `S_E = 0.4 Seco + 0.4 Scarbon + 0.2 Sdisturb` | `model/evaluation.py:evaluate_unit` | `eco_score`, `carbon_level`, `disturbance_level` | 已实现；陡坡额外扣减扰动分 | 基本一致；陡坡扰动扣减为当前规则，待标定 |
| `SR = 0.5 Sredundancy + 0.3 Srecover + 0.2 Scoverage` | `model/evaluation.py:evaluate_unit` 与 `model/regional.py:evaluate_region` | `function_count`, `recovery_difficulty`, `covered_paths` / `path_coverage` | 单元层先算 `resilience`；区域层若存在拓扑，则改用路径覆盖重算区域韧性 | 基本一致；区域层已把路径覆盖纳入优化前评价 |
| `Sredundancy = min(100, 40 + 15 Nf)` | `model/evaluation.py:evaluate_unit` | `function_count` | 有措施时按 `40 + 15*Nf`；无措施时用 15 起点 | 部分一致；无措施基线为当前规则 |
| `Srecover = 100 − 20 (Mp − 1)` | `model/evaluation.py:evaluate_unit` | `recovery_difficulty` | 已实现 | 一致 |
| `Scoverage = 100 × Ncovered / Nkey` | `model/evaluation.py:evaluate_unit`, `model/spatial.py:evaluate_path_coverage` | `covered_paths`, `key_paths` 或 `topology.paths` | 单元局部覆盖与区域关键路径覆盖均已实现 | 一致 |
| `Sregion = 0.6 min(Si) + 0.4 Σ(wi·Si)` | `model/regional.py:evaluate_region` | 单元安全性、`risk_weight` | 已实现区域安全性 | 一致 |
| 区域经济/可持续/韧性按风险权重加权汇总 | `model/regional.py:evaluate_region` | 单元 `economy/sustainability/resilience` | 已实现风险权重加权平均 | 一致 |
| 地形地貌与工程适宜性影响准入、四目标评价和区域协同 | `model/screening.py`, `model/evaluation.py`, `model/regional.py` | 坡度、坡高、可达性、关键路径 | 已部分实现 | 部分一致；仍有正式映射待补充 |

## 3. 关键参数来源与状态

| 参数/符号 | 当前来源 | 默认值/实现 | 状态 |
| --- | --- | --- | --- |
| `FS0`, `Krain`, `risk_weight`, `key_paths` | `data/default_units.csv` | 随样例数据载入 | 开发示例/待标定 |
| `ΔFSp`, `chain_delta`, `build_level`, `maintain_level`, `construct_level` | `data/default_measures.csv` | 随样例数据载入 | 开发示例/待标定 |
| `drainage_synergy` | `data/default_scenario.json` / UI 可调 | `1.10` | 平台当前规则/待标定 |
| `general_second_measure_factor` | `data/default_scenario.json` / UI 可调 | `0.90` | 平台当前规则/待标定 |
| `SFS` 线性映射区间 | `model/parameters.py:EvaluationParameters` | `FS=0.80 -> 0`, `FS=1.40 -> 100` | 平台当前规则/待标定 |
| 陡坡扰动扣减 | `model/parameters.py:EvaluationParameters` | `slope_deg > 45` 时 `disturb -= 5` | 平台当前规则/待标定 |
| 货币化维护/组织成本折现 | `model/parameters.py:EvaluationParameters` | `0.03 + 0.015*(maintain_level-1)`，`0.02*(construct_level-1)` | 平台当前规则/待标定 |

## 4. 当前已确认的实现结论

- 区域协同没有被放到优化后单独首次计算；它已经进入 `evaluate_region()` 和 `evaluate_path_coverage()`，因此会影响候选方案的区域韧性、路径覆盖与支配关系。
- 精确枚举与 NSGA-II 当前共用 `evaluate_village_scheme()`，这一点与方法要求一致。
- `coordination_metrics()` 仍主要用于页面解释与导出，不是优化主评价入口；优化入口使用的是 `evaluate_path_coverage()`。
- 经济性当前采用双轨口径：
  - `economy = lci_score` 进入四目标优化；
  - `lifecycle_cost_total` 用于预算约束、展示和导出。

## 5. 仍待补充正式依据的内容

- `risk_weight` 的完整生成公式尚未在正式材料中提取到可直接编码的版本。
- `SFS` 从 `FSrain` 到 0–100 分的线性映射区间尚属平台当前规则。
- 多措施叠加的协同与折减规则尚属平台当前规则。
- 陡坡扰动扣减、维护折现和组织成本换算尚属平台当前规则。
- 关键路径自动识别与“已覆盖/已阻断”的精细判定仍依赖当前示例拓扑数据结构。

## 6. V1空间输入与参数溯源补充（2026-09-15）

- 正式入口改为 `v1_app.py` / `api.py`，两者调用 `model/service.py`。`data_adapter.py` 读取有空间参照的评价成果，`governance.py` 根据报告表5-1识别重点区域并生成单元；不要求上游预先划好治理单元。
- 单元风险权重必须显式提供，负值、非有限值及全零权重被拒绝；程序仅归一化，不推定风险权重公式。
- 存在topology时，单元关键路径数量和局部覆盖直接来自候选方案实际路径记录；村域统计完整路径覆盖，二者没有分别维护人工路径计数。topology进入输入标识，改变后原结果失效。
- 原评价中的待标定常数集中到 `EvaluationParameters`，原默认值及正式指标权重不变。配置在输入、输出与成果包中完整留存。空间规则集中于 `SpatialConfig`。
- 当前V1模拟空间数据、工程调查属性及人工路径见 `data/v1/sample_request.json`。真实数据不足和待标定边界、实测求解对照与验收结果见 `V1_RUN_REPORT.md`。
