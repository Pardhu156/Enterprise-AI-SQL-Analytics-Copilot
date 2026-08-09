from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frontend_modules_import_without_postgresql_driver() -> None:
    code = """
import importlib.abc
import sys

class BlockPsycopg(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'psycopg2' or fullname.startswith('psycopg2.'):
            raise ModuleNotFoundError("psycopg2 intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockPsycopg())

from src.analytics.chart_selector import ChartConfig
from src.analytics.visualization import VisualizationEngine
from src.api.schemas.responses import AnalyticsQueryResponse
from src.frontend.api_client import AnalyticsAPIClient

assert 'src.analytics.analytics_pipeline' not in sys.modules
assert 'src.text_to_sql.pipeline' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
