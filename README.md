# 村域综合整治工程技术智能研判与决策平台

**当前交付为 V1.2。** 优先查看 [输入资料与输出清单](docs/INTEGRATION_DATA_GUIDE.md)、[本轮改进与验收报告](docs/V1_2_RUN_REPORT.md)。独立入口为 `http://127.0.0.1:8017/integration`：内置资料自动载入，逐项校验、替换后计算；结果按“治理方案 / 治理成效 / 计算溯源”展示。入口不依赖 Streamlit，`v1_app.py` 同步提供逐项资料清单与共用地图。

- 可交接输入模板：[peitian_input_package.zip](data/v1_1/peitian_input_package.zip)，分别包含GeoTIFF、研究范围、调查CSV及空间权重/路径，无预定义治理单元。
- 本轮真实底图与工程图标地图：[spatial_decision_map.html](output/v1_1/spatial_decision_map.html)。在线影像需要网络，计算图层与符号在文件内。
- 本轮完整成果：[decision_results.zip](output/v1_1/decision_results.zip)。包含措施点GeoJSON及八类SVG符号，供宿主系统直接调用。
- 本轮批处理：`.venv_codex\Scripts\python.exe -X utf8 run_v1.py --input data/v1_1/peitian_request.json --output output/v1_1 --compare`。

以下保留V1基础版本说明和回归样例路径；V1.1新增内容及当前数据请以上述资料清单为准。

当前 V1 已接通“空间评价成果 → 重点治理区域 → 自动治理单元 → 工程属性与措施筛选 → 统一四目标及区域协同 → 多目标优化 → 代表方案 → 空间治理地图”。复用原 V4 方法内核、八类工程措施和离散 NSGA-II（第二代非支配排序遗传算法，用于近似搜索多目标非劣方案），保留精确枚举基准与原开发工作台。

当前样例全部是明确标记的 MOCK 模拟空间评价数据和人工路径，**不是实际村庄调查或正式工程研究成果**。已确认公式遵循 `docs/METHOD_SPEC.md`；工程效应、费用、组合规则和空间划分细则仍需标定。

## 快速查看本轮结果

- 完整运行报告：[docs/V1_RUN_REPORT.md](docs/V1_RUN_REPORT.md)
- 独立交互治理地图：[output/v1/map.html](output/v1/map.html)，可直接双击打开，无需网络或Python。
- 完整数据与成果包：[output/v1/decision_results.zip](output/v1/decision_results.zip)
- 实际输入：[data/v1/sample_request.json](data/v1/sample_request.json)
- 集成接口：[docs/V1_API.md](docs/V1_API.md)
- 系统设计与审计：[docs/V1_SYSTEM_DESIGN.md](docs/V1_SYSTEM_DESIGN.md)
- 实施计划：[docs/V1_IMPLEMENTATION_PLAN.md](docs/V1_IMPLEMENTATION_PLAN.md)

## Windows 启动

建议使用64位 Python 3.12。已有 `.venv_codex` 可直接运行：

```powershell
.venv_codex\Scripts\python.exe -m pip install -r requirements.txt
.venv_codex\Scripts\python.exe -m streamlit run v1_app.py --server.address 127.0.0.1 --server.port 8517
```

访问 `http://127.0.0.1:8517`。也可以双击 `start_windows.bat`（端口8513）或运行 `start_windows.ps1`（默认8501）；两者使用 `.venv`，缺少新增 GIS/API 依赖时会安装。原 `app.py` / `ui_v3.py` 保留为历史开发工作台，不是新正式入口。

## 正式页面操作

1. 选择“模拟示例空间数据”，或上传符合接口规范的项目 JSON。
2. 页面读取基础评价图层，显示有效片区数、重点面积和自动生成单元数；必要时调整预算及求解方式。
3. 点击“运行综合研判”，等待实际计算完成。
4. 切换综合均衡、安全、经济、韧性、可持续五类方案，地图、费用和指标同步变化。
5. 点击单元面或编号点，查看工程措施、原风险、前后安全/韧性、成本及复核事项。
6. 查看四目标雷达、Pareto（帕累托非劣，即无法在不降低其他目标的同时提高某一目标）投影、措施组成；下载空间数据或成果包。
7. “开发调试”模式展开单元、候选、实际迭代和参数来源。输入变化时原结果失效，上传失败不会回退使用示例。

## 完整批处理与对照

```powershell
.venv_codex\Scripts\python.exe -X utf8 run_v1.py --compare
```

结果写入 `output/v1`。可通过 `--input 项目.json --output 输出目录` 替换数据。`--compare` 实际运行精确枚举和三个种子 NSGA-II；未指定时只运行请求中的求解方法。

本轮144个模拟评价像元形成24个重点像元、4个治理单元。精确枚举4,356种组合，407可行、16非劣；NSGA-II在42、123、2026三个种子下完整找回这16个非劣方案。该结论仅对应本例及当前参数。

## API 集成

```powershell
.venv_codex\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8017
```

接口文档：`http://127.0.0.1:8017/docs`。接口为 `GET /health`、`POST /prepare`、`POST /optimize`。上游提供村界、空间评价数据、工程调查属性、显式权重、路径和配置，返回单元、完整非劣集、代表方案、GeoJSON地图和溯源信息。服务和页面调用同一个 `model.service.optimize_request`。

支持带 CRS 的 vector、栅格数值网格、多波段 GeoTIFF；本地适配支持 GeoJSON、CSV/WKT、XLSX/WKT。普通图片不是 GIS。栅格需同网格，矢量需无重叠；未提供自动重采样、Shapefile直读或无依据的风险权重/关键路径推断。

## 验证

```powershell
.venv_codex\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv_codex\Scripts\python.exe -m compileall -q model api.py v1_app.py run_v1.py scripts tests
```

完整测试、启动、API、浏览器点击与上传核验的本轮结果见运行报告及 `output/v1` 日志/截图。旧 `sample_output` 保留原V3结果，不能与本轮混用。

## 科研边界

四目标保持正式方法：经济性采用 LCI/SC，货币总成本仅用于预算与金额输出；区域协同在每个候选方案评价中参与四目标形成。风险权重显式输入并归一化，路径展示与计算共用 topology。连通分区、面积过滤/拆分、组合效应、费用及工程参数尚需真实村庄资料标定。措施费用按每单元整套估算，尚无工程量清单，细分单元会改变费用；不能将当前推荐直接当作施工设计。
