"""
Tests for controllers/agent.py
Run: pytest tests/test_agent.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def _mock_download(ticker, period="5y", progress=False, auto_adjust=True, **kw):
    np.random.seed(hash(ticker) % 2**31)
    dates = pd.bdate_range("2020-01-01", periods=1000)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 1000)))
    return pd.DataFrame({"Close": prices}, index=dates)


@pytest.fixture
def portfolio(tmp_path):
    from models.portfolio import Portfolio
    csv = tmp_path / "test.csv"
    csv.write_text(
        "ticker,name,sector,asset_class,quantity,purchase_price\n"
        "AAPL,Apple,Technology,Equity,10,150.00\n"
        "BND,Bond ETF,Fixed Income,Bond,20,75.00\n"
    )
    with patch("yfinance.download", side_effect=_mock_download):
        return Portfolio.from_csv(csv)


@pytest.fixture
def controller(portfolio):
    from controllers.agent import PortfolioController
    return PortfolioController(portfolio, api_key="test-key")


class TestToolExecution:
    def test_overview_returns_string(self, controller):
        with patch("yfinance.download", side_effect=_mock_download):
            result = controller._execute_tool("show_portfolio_overview", {})
        assert isinstance(result, str)
        assert "AAPL" in result

    def test_weights_returns_string(self, controller):
        result = controller._execute_tool("show_weights", {"by": "all"})
        assert isinstance(result, str)
        assert "%" in result

    def test_risk_returns_data(self, controller):
        with patch("yfinance.download", side_effect=_mock_download):
            result = controller._execute_tool("show_risk_metrics", {"level": "both"})
        assert isinstance(result, str)
        assert "Sharpe" in result

    def test_unknown_tool_returns_error(self, controller):
        result = controller._execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_concentration_returns_string(self, controller):
        result = controller._execute_tool("show_concentration_risk", {})
        assert isinstance(result, str)

    def test_stress_test_returns_string(self, controller):
        with patch("yfinance.download", side_effect=_mock_download):
            result = controller._execute_tool("run_stress_test", {})
        assert isinstance(result, str)

    def test_tool_handles_exception(self, controller):
        """Tool should return error string, not raise."""
        with patch.object(controller.portfolio, "all_risk_metrics", side_effect=Exception("test error")):
            result = controller._execute_tool("show_risk_metrics", {"level": "asset"})
        assert "Error" in result or "error" in result
