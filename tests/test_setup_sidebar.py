import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.helper import _setup_sidebar


def test_generate_streamlit_sidebar() -> None:
    assert _setup_sidebar() is None
