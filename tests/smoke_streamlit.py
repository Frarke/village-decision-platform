"""第三版Streamlit首屏真实脚本运行检查。"""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
if app.exception:
    raise RuntimeError("；".join(str(item.value) for item in app.exception))
visible = "\n".join(str(item.value) for item in app.markdown)
if "综合整治工程技术多目标优化智能研判与决策平台" not in visible:
    raise RuntimeError("第三版正式标题未渲染。")
if not app.radio or list(app.radio[0].options) != [
    "阶段一 治理对象与基础数据",
    "阶段二 多维综合评估",
    "阶段三 多目标优化求解",
    "阶段四 方案评审与决策输出",
]:
    raise RuntimeError("四个一级阶段未按预期渲染。")
if any(term in visible for term in ["MVP", "Demo版本", "AI生成"]):
    raise RuntimeError("前台包含不应出现的开发定位表述。")
print("Streamlit首屏检查通过：0个异常，四个一级阶段与主工作区成功渲染。")
