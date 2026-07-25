"""第三版Streamlit首屏真实脚本运行检查。"""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
if app.exception:
    raise RuntimeError("；".join(str(item.value) for item in app.exception))
visible = "\n".join(str(item.value) for item in app.markdown)
if "GIS智能研判与多目标决策平台" not in visible:
    raise RuntimeError("第三版正式标题未渲染。")
if not app.radio or list(app.radio[0].options) != ["项目总览", "数据与单元", "智能研判", "方案比选", "空间协同", "成果管理"]:
    raise RuntimeError("六个一级模块未按预期渲染。")
if any(term in visible for term in ["MVP", "Demo版本", "AI生成"]):
    raise RuntimeError("前台包含不应出现的开发定位表述。")
print("Streamlit第三版首屏检查通过：0个异常，六个一级模块与GIS主工作区成功渲染。")
