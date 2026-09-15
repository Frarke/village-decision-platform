from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_formal_run_switch_unit_switch_plan_and_stale_results():
    app = AppTest.from_file(str(ROOT / "v1_app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].value == "144"
    next(b for b in app.button if b.key == "v1_run").click().run()
    assert not app.exception
    original_result = app.session_state["v1_result"]
    radio = next(r for r in app.radio if r.key == "v1_plan_label")
    assert len(radio.options) == 5
    balanced_cost = app.metric[0].value
    radio.set_value("经济优先").run()
    assert not app.exception
    assert app.metric[0].value != balanced_cost
    next(s for s in app.selectbox if s.key == "v1_selected_unit").set_value("U03").run()
    assert not app.exception
    assert app.session_state["v1_selected_unit"] == "U03"
    assert app.session_state["v1_result"]["input_hash"] == original_result["input_hash"]
    next(r for r in app.radio if r.key == "v1_mode").set_value("开发调试").run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["单元与措施", "真实迭代", "输入与依据"]
    app.number_input[0].set_value(400.0).run()
    assert not app.exception
    assert "全寿命成本（万元）" not in [m.label for m in app.metric]
    assert any("已失效" in w.value for w in app.warning)


def test_upload_mode_does_not_reuse_sample_when_no_file():
    app = AppTest.from_file(str(ROOT / "v1_app.py"), default_timeout=60).run()
    next(s for s in app.selectbox if s.key == "v1_source").set_value("上传项目数据").run()
    assert not app.exception
    assert not any(b.key == "v1_run" for b in app.button)
