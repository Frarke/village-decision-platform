# V1 数据与集成接口

启动：`.venv_codex\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8017`。接口说明：`http://127.0.0.1:8017/docs`。标准环境也可使用安装了 requirements.txt 的 `.venv`。

这是有规模限制的同步本地服务。正式部署前需接入宿主平台的鉴权、请求限流和异步作业机制；本轮不引入账户/数据库，也不开放公网端口。

## 对方平台需要提供什么

|字段|含义和要求|
|---|---|
|schema_version|字符串 `1.0`|
|village|`id`、`name`、`source`、`data_status`（REAL/MOCK）、`crs`、面几何 `boundary`|
|spatial_layers|`kind` 为 vector / raster_grid / geotiff，格式见下文|
|engineering_profiles|工程调查属性区字典，键与基础片区的 `profile_id` 对应|
|unit_risk_weights|按自动生成的 U01… 显式赋权，有限非负且总和>0；程序只归一化|
|topology|显式 `source`、`data_status`、`paths`；路径条目含 path_id/name/source_unit/transport_unit/receptor_unit，key为可选布尔值，默认关键路径|
|configuration|`method` 为 exact/nsga2；`spatial` 为空间规则；`solver` 为预算、安全底线、算法和评价参数|
|measures|可选。未传时使用内置八类待标定措施；传入时按 Measure 字段校验并真实替换|

上游无需预先划定治理单元，但须提供具有共同空间参照的评价成果和必要调查条件。不能仅提供一张 PNG/JPG。易发性/易损性不自动替代工程稳定性、施工条件和风险权重。示例请求见 `data/v1/sample_request.json`（全部模拟数据）。

## 基础评价字段

每个基础面/像元必须有：`risk`、`haz`（1～5整数等级）、`distance_m`（距承灾体、非负米）、`slope_deg`（0～90度）、`profile_id`。可选：`susceptibility`、`vulnerability`（0～1），以及已有 `identified_or_risk_area`、`focus_unit` 布尔标记。

可选评价图层须覆盖全部有效片区或整体省略。任何缺失值不得解释为0风险；栅格任一已提供波段为NoData时，该像元从统计中排除，计数和有效覆盖面积回显。数据不完整时只能对有效覆盖范围作判断，不能声称识别了全村全部风险。

### 矢量

```json
{
  "kind": "vector", "crs": "EPSG:4326",
  "features": [{
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [[[116.09,25.10],[116.091,25.10],[116.091,25.101],[116.09,25.101],[116.09,25.10]]]},
    "properties": {"feature_id":"F01", "profile_id":"1", "risk":3, "haz":3, "distance_m":80, "slope_deg":32, "susceptibility":0.7, "vulnerability":0.6}
  }]
}
```

`feature_id`非空唯一；面必须有效，无面积重叠，保留洞和多面。MultiPolygon内不相连部分分别参与连通识别。矢量输入是已叠置的基础评价面；当前不在服务中自动猜测多图层关联字段。Shapefile等可先按显式CRS转换为此格式，当前未提供Shapefile直读。

### 数值栅格

`kind=raster_grid`；`crs`；`transform=[a,b,c,d,e,f]`，满足 `x=a*col+b*row+c, y=d*col+e*row+f`，坐标对应像元角点；`layers`是上述字段的同尺寸二维数组。`profile_id`可为整数编码，调查字典用同值字符串键。`nodata`可选，JSON null也代表缺失。不同分辨率、范围或行列的栅格须上游先对齐；程序不静默重采样。

### GeoTIFF

`kind=geotiff`，`content_base64`（文件内容Base64）、`band_fields`（按1起始波段顺序列出字段名）。服务直接读取GeoTIFF的CRS、仿射变换和NoData，再走同一流程。不接收服务器本地路径或任意远程地址。单文件≤20MB、基础片区/像元≤10,000。示例实物见 `output/v1/MOCK_spatial_evaluations.tif`，波段表见同目录 `raster_band_fields.json`。

本地 Python 适配入口 `model.data_adapter.read_spatial_file(path, crs=..., band_fields=...)` 支持 GeoJSON、GeoTIFF、CSV/WKT、XLSX/WKT。属性表须包含 `geometry_wkt` 列和明确CRS；普通GeoJSON按RFC7946默认WGS84，可显式覆盖并核对。Shapefile、独立图片、无几何CSV返回说明，不能偷偷作为空间数据。

## 工程调查属性

`engineering_profiles` 的每条记录必须显式提供：

- 数值：`rainfall_mm`、`height_m`、`catchment_km2`、`gully_level`、`foot_space_m`、`fs0`、`k_rain`。
- 布尔：`safe_drainage_outlet`、`artificial_or_bare`、`anchor_condition`、`slip_surface_or_review`、`slip_surface_confirmed`、`motion_path`、`accessible`、`wall_back_drainage`。
- 可选说明：`unit_name`、`evidence_level`、`notes`。

这些字段来自原准入/评价器的实际需求，不由易发性或图片猜测。风险、危险性、坡度、距离、面积和位置由空间片区提取，调查属性不得覆盖这些自动字段。属性区只是调查信息归属，不要求提前给出最终治理单元。

## 先生成单元，再赋权（真实接入推荐）

1. `POST /prepare`：提供村界、空间评价数据及空间配置；返回 `governance_units`、`priority_areas`、`required_risk_weight_ids`、`spatial_metrics`。
2. 对返回编号输入 `unit_risk_weights`，配置经复核的关键路径。若已确认没有关键路径，显式设置 `paths=[]` 且 `status=confirmed_empty`。当前沿用已记录的空路径覆盖100分暂定约定，响应会提示。
3. `POST /optimize`：同一空间输入、配置、工程调查属性、显式权重和路径，得到完整结果。已知编号的集成项目也可一次调用。

改变最小面积、拆分规则或评价数据后编号可能变化，必须重新核对权重和路径。不要把旧编号盲目套用到新划分。输入摘要包含整个请求；返回provenance同时记录实际措施库、全部默认展开配置和来源。

## 配置与限制

|键|默认值|依据/状态|
|---|---:|---|
|spatial.risk_threshold|3|科技报告表5-1，决策级阈值|
|spatial.receptor_distance_m|100|科技报告表5-1|
|spatial.min_area_m2|100|V1暂定小斑块过滤|
|spatial.merge_distance_m|0|V1暂定，共边连通；角接触分开|
|spatial.max_unit_area_m2|null|V1暂定可选网格拆分，默认关闭|
|spatial.max_units|64|同步服务资源限制|
|solver.total_budget|520万元|示例预算，真实项目自行提供|
|solver.high_risk_min_fs / high_risk_min_safety|1.05 / 58|待标定安全底线|
|solver.max_measures_per_unit|2|组合配置|
|solver.nsga_population_size / nsga_generations|60 / 25|当前算法设置，不是正式最优参数|
|solver.random_seed|42|固定可复现|

其余字段见 `model.schemas.SolverConfig`。评分映射、组合外参数、成本地形修正等可通过 `solver.evaluation_parameters` 配置，字段与默认值见 `model.parameters.EvaluationParameters`，全部展开后保存在结果中。正式四目标权重不从此入口擅自改写。

## 返回结构

- `recommended_solution`：综合均衡方案；`representative_solutions`：五种偏好、完整目标、成本、单元详情、路径状态、选择依据、约束状态。
- `pareto_front`：完整本次搜索非劣集，保留四目标、成本及按生成单元顺序的措施编码。精确枚举是完整前沿；NSGA-II是近似搜索结果。
- `governance_units`：WGS84 FeatureCollection，含面积、均值/最大值、风险、距离、位置、来源片区、工程属性、候选措施、候选组合、输入权重和归一化权重。
- `priority_areas`：阈值命中的基础空间区域，保留判定原因；包括可能被最小面积规则过滤的斑块。
- `map_result.solutions[偏好名称]`：各方案的WGS84空间结果，含geometry、措施、原风险、前后安全/韧性、成本、复核项。
- `summary`、`metrics`：真实计数、算法历史、空间处理统计、基线和依据提示。
- `provenance`：输入村庄、路径、措施、参数版本和依据。MOCK/REAL含义由来源声明，REAL标记本身不等于已通过科研验收。

没有重点区域或全部被面积规则过滤时，返回 `status=no_governance_units`、空方案集和明确统计；无可行方案则返回422和预算/安全差距，不虚构推荐。未知程序异常记录日志并返回500。

## 调用示例

```powershell
$payload = Get-Content data\v1\sample_request.json -Raw -Encoding utf8
$result = Invoke-RestMethod -Uri http://127.0.0.1:8017/optimize -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
$result.summary
```

无HTTP也可调用 `model.service.optimize_request(request_dict)`，与网页及接口是同一服务。完整流程与文件输出：`.venv_codex\Scripts\python.exe -X utf8 run_v1.py --input data\v1\sample_request.json --output output\v1 --compare`。
