from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_external_calls() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert any(
        "Enterprise AI SQL Analytics Copilot" in markdown.value
        for markdown in app.markdown
    )
    assert any(button.label == "Analyze" for button in app.button)
