"""Streamlit 首页基础冒烟检查。"""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()

if app.exception:
    raise RuntimeError("；".join(str(item.value) for item in app.exception))

visible = "\n".join(str(item.value) for item in app.markdown)
button_labels = [str(item.label) for item in app.button]
if "综合整治工程技术多目标优化智能研判与决策平台" not in visible:
    raise RuntimeError("首页正式标题未按预期渲染。")

if len(app.radio) < 1:
    raise RuntimeError("首页缺少工作模式控件。")

if list(app.radio[0].options) != ["平台首页", "引导式研判", "快速分析", "算法验证中心"]:
    raise RuntimeError("工作模式选项未按预期渲染。")

for term in ["四阶段正式计算框架", "工程措施库概览"]:
    if term not in visible:
        raise RuntimeError(f"首页缺少关键展示内容：{term}")

for label in ["进入快速分析", "进入引导式研判"]:
    if label not in button_labels:
        raise RuntimeError(f"首页缺少关键入口按钮：{label}")

if any(term in visible for term in ["MVP", "Demo版本", "AI生成"]):
    raise RuntimeError("前台包含不应出现的开发定位表述。")

print("Streamlit 首页检查通过：首页入口、四阶段框架和措施库概览均已渲染。")
