# 项目背景与上下文

## 1. 文档目的

本文件用于沉淀“综合整治工程技术多目标优化智能研判与决策平台”第一轮正式方法梳理的项目背景、研究边界、材料来源和当前实现定位，作为后续 `METHOD_SPEC.md`、`IMPLEMENTATION_AUDIT.md`、`DECISIONS.md` 和 `V4_PLAN.md` 的上位背景说明。

## 2. 研究任务与项目定位

- 本项目面向东南降雨型群发滑坡村域尺度综合整治问题，目标是在“治理单元确定、措施适用性筛选、四目标评价、区域协同评价、多目标优化、方案评审与输出”之间建立统一决策链条。[科技报告明确] 来源：`references_private/tech_report/中期科技报告.docx`，综合整治工程技术决策框架、综合整治工程技术比选方法。
- 所属专题任务为“多目标优化的村庄地质灾害综合整治工程技术智能研判与决策模型”，要求综合评估安全性、经济性、可持续性和韧性水平等多因素，构建多目标优化智能研判与决策模型。[任务书明确] 来源：`references_private/task_requirements/任务说明.docx` 第9-12段。
- 中期阶段正式成果侧重于“构建综合整治工程技术决策框架，提出综合整治工程技术比选方法”，并形成可接入工程目标函数的 NSGA-II 优化接口与平台承载环境。[科技报告明确] 来源：`references_private/tech_report/中期科技报告.docx` 第3-5段。

## 3. 正式材料清单与状态

### 3.1 已成功读取的正式材料

- `references_private/tech_report/中期科技报告.docx`。[科技报告明确]
- `references_private/task_requirements/任务说明.docx`。[任务书明确]
- `references_private/midterm_slides/中期PPT.pptx`。[PPT明确]
- `references_private/matlab_original/00Nsga-ii 批处理与算法内核 Readme（打包版）.docx`。[MATLAB代码现状]
- `references_private/matlab_original/00nsga_ii_批处理与算法内核_readme（打包版）.md`。[MATLAB代码现状]
- `references_private/matlab_original/*.m` 全部 MATLAB 源文件（`evaluate_objective.m`、`initialize_variables.m`、`genetic_operator.m`、`non_domination_sort_mod.m`、`tournament_selection.m`、`replace_chromosome.m`、`repair_offspring.m`、`nsga2_core.m`、`nsga2_batch.m`、`Main.m`、`find_pareto_front.m`、`NSGA_ALL2.m`、`plot_convergence_metrics.m`、`plot_pareto_front_3D.m`）。[MATLAB代码现状]

### 3.2 已成功读取的 V3 实现与支撑材料

- 根目录：`README.md`、`STATUS.md`、`requirements.txt`、`app.py`、`ui_v3.py`。
- 模型目录：`model/*.py` 全部源码。
- 数据目录：`data/default_units.csv`、`data/default_measures.csv`、`data/default_rules.csv`、`data/assumptions.csv`、`data/parameter_catalog.csv`、`data/spatial_units.geojson`、`data/spatial_context.geojson`、`data/spatial_topology.json`。
- 测试目录：`tests/test_acceptance.py`、`tests/test_round2.py`、`tests/test_round3.py`、`tests/smoke_streamlit.py`、`tests/check_streamlit_step.py`。
- 既有说明文档：`docs/audit_report.md`、`docs/model_definition.md`、`docs/model_mapping.md`、`docs/validation_report.md` 等。

### 3.3 未作为主要依据的内容

- `sample_output/*`：为示例输出与验证产物，不作为正式方法依据。[开发示例]
- `STATUS.md`、既有 `docs/*.md`：属于 V3 开发记录或说明，不高于科技报告、任务书和用户确认结论。[V3代码现状]

## 4. 当前 V3 的实现定位

- V3 已将“治理单元—措施库—准入筛选—四目标评价—精确枚举—离散 NSGA-II—代表方案—空间协同—成果导出”串联为可运行流程。[V3代码现状] 来源：`README.md`、`ui_v3.py`、`model/solver.py`、`model/nsga2.py`。
- V3 当前使用的空间单元、GeoJSON 边界和风险路径均为示例数据，不代表现场调查成果。[开发示例] 来源：`README.md`、`data/spatial_topology.json`、`data/default_units.csv`、`data/assumptions.csv`。
- V3 中大量效应值、阈值、成本修正和评分映射仍属于待标定参数，不能表述为正式工程定值。[待标定] 来源：`data/assumptions.csv`、`data/parameter_catalog.csv`、`model/evaluation.py` 注释。

## 5. 当前阶段应遵循的研究边界

- 标准测试函数 ZDT1/ZDT2/ZDT3/DTLZ2 仅能用于算法机制验证，不得直接表述为工程优化问题已经完成验证。[科技报告明确][用户已确认] 来源：`references_private/tech_report/中期科技报告.docx`“标准函数验证”段；当前任务要求。
- MATLAB 原始 NSGA-II 原型与工程四目标模型尚未完成一体化，不能因为存在“可接入接口”就表述为工程目标已在 MATLAB 中实现。[MATLAB代码现状][待用户确认] 来源：`evaluate_objective.m`、`nsga2_core.m`、科技报告“多目标优化算法搭建”段。
- 区域协同评价在正式方法中属于单元评价结果向村域方案评价转换的关键环节，后续版本需确保其进入权威候选方案评价链，而不只是页面展示指标。[科技报告明确][待用户确认] 来源：科技报告“基于风险权重的区域协同评价”。
- 正式方法需要区分“科研方法流程”“优化算法内部评价流程”“网页展示流程”三层，不得把页面步骤顺序误写成后台真实计算顺序。[用户已确认]
- MATLAB 中 `A-G`、`omega`、`FS` 等历史试验公式，在缺少完整变量定义、公式来源和正式研究依据前，不进入 V4 正式工程评价体系。[用户已确认]

## 6. 本轮文档产出原则

- 一切正式结论优先引用科技报告、任务书和 PPT。
- 涉及 MATLAB 或 V3 的判断必须明确标注“现状”，不得上升为正式研究结论。
- 涉及示例 GeoJSON、示例单元、示例路径、示例参数和待标定参数时，必须显式标识。
