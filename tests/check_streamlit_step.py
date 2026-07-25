"""独立进程检查六个一级模块，避免页面状态相互污染。"""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


modules = ["项目总览", "数据与单元", "智能研判", "方案比选", "空间协同", "成果管理"]
index = int(sys.argv[1])
root = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(root / "app.py"), default_timeout=60)
app.session_state["current_module"] = modules[index]
app.session_state["module_nav"] = modules[index]
app.session_state["current_step"] = [0, 1, 2, 5, 6, 7][index]
app.run(timeout=60)
if app.exception:
    raise RuntimeError("；".join(str(item.value) for item in app.exception))
if app.radio[0].value != modules[index]:
    raise RuntimeError(f"一级模块未切换到{modules[index]}。")
print(f"模块{index + 1}（{modules[index]}）页面检查通过。")
