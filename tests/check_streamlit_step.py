"""独立进程检查四个一级阶段入口，避免页面状态互相污染。"""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest


MODULES = [
    "阶段一 治理对象与基础数据",
    "阶段二 多维综合评估",
    "阶段三 多目标优化求解",
    "阶段四 方案评审与决策输出",
]

STEPS = [0, 2, 4, 6]
PAGE_IDS = [f"guided_step_{step}" for step in STEPS]

index = int(sys.argv[1])
root = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(root / "app.py"), default_timeout=60)
app.session_state["mode"] = "引导式研判"
app.session_state["current_page_id"] = PAGE_IDS[index]
app.session_state["pending_page_id"] = None
app.session_state["current_module"] = MODULES[index]
app.session_state["current_step"] = STEPS[index]
app.run(timeout=60)

if app.exception:
    raise RuntimeError("；".join(str(item.value) for item in app.exception))

if len(app.radio) < 2:
    raise RuntimeError("页面缺少工作模式或一级阶段导航。")

if list(app.radio[0].options) != ["平台首页", "引导式研判", "快速分析", "算法验证中心"]:
    raise RuntimeError("工作模式控件未按预期渲染。")

if app.radio[1].value != MODULES[index]:
    raise RuntimeError(f"一级模块未切换到 {MODULES[index]}。")

print(f"模块 {index + 1}（{MODULES[index]}）页面检查通过。")
