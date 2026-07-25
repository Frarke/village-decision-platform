"""Streamlit入口。界面实现集中在ui_v3，便于状态机与模型代码独立测试。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_v3 import main


main()
