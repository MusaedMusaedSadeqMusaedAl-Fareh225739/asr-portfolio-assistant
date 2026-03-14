"""
Tests for models/portfolio.py
Run: pytest tests/ -v --cov=models
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_download(ticker, period="5y", progress=False, auto_adjust=True, **kw):
    """Generate fake price data for any ticker."""
    np.random.seed(hash(ticker) % 2**31)
    dates = pd.bdate_range("2020-01-01", periods=1000)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 1000)))
    df = pd.DataFrame({"Close": prices}, index=dates)
    return df


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV portfolio file."""
    csv = tmp_path / "test_portfolio.csv"
    csv.write_text(
        "ticker,name,sector,asset_class,quantity,purchase_price\n"
        "AAPL,Apple,Technology,Equity,10,150.00\n"
        "MSFT,Microsoft,Technology,Equity,5,280.00\n"
        "JPM,JPMorgan,Financials,Equity,8,130.00\n"
        "BND,Vanguard Bond,Fixed Income,Bond,20,75.00\n"
        "GLD,Gold ETF,Commodities,Commodity,3,170.00\n"
    )
    return csv


@pytest.fixture
def portfolio(sample_csv):
    """Load a portfolio with mocked yfinance."""
    from models.portfolio import Portfolio
    with patch("yfinance.download", side_effect=_mock_download):
        return Portfolio.from_csv(sample_csv)


# ── Loading Tests ─────────────────────────────────────────────────────────────

class TestLoading:
    def test_csv_loads_correct_count(self, portfolio):
        assert len(portfolio.assets) == 5

    def test_csv_tickers_uppercase(self, portfolio):
        for a in portfolio.assets:
            assert a.ticker == a.ticker.upper()

    def test_csv_missing_columns_raises(self, tmp_path):
        from models.portfolio import Portfolio
        bad = tmp_path / "bad.csv"
        bad.write_text("ticker,name\nAAPL,Apple\n")
        with pytest.raises(ValueError, match="missing columns"):
            Portfolio.from_csv(bad)

    def test_json_loads(self, tmp_path):
        from models.portfolio import Portfolio
        j = tmp_path / "test.json"
        j.write_text('[{"ticker":"AAPL","name":"Apple","sector":"Tech","asset_class":"Equity","quantity":10,"purchase_price":150}]')
        with patch("yfinance.download", side_effect=_mock_download):
            p = Portfolio.from_json(j)
        assert len(p.assets) == 1


# ── Portfolio Properties ──────────────────────────────────────────────────────

class TestProperties:
    def test_total_cost(self, portfolio):
        expected = 10*150 + 5*280 + 8*130 + 20*75 + 3*170
        assert portfolio.total_cost == expected

    def test_total_value_positive(self, portfolio):
        assert portfolio.total_value > 0

    def test_total_pnl_type(self, portfolio):
        assert isinstance(portfolio.total_pnl, float)

    def test_pnl_pct_calculation(self, portfolio):
        expected = portfolio.total_pnl / portfolio.total_cost * 100
        assert abs(portfolio.total_pnl_pct - expected) < 0.01


# ── Weights ───────────────────────────────────────────────────────────────────

class TestWeights:
    def test_weights_by_asset_sum_100(self, portfolio):
        df = portfolio.weights_by_asset()
        assert abs(df["Weight (%)"].sum() - 100) < 0.01

    def test_weights_by_sector_sum_100(self, portfolio):
        df = portfolio.weights_by_group("sector")
        assert abs(df["Weight (%)"].sum() - 100) < 0.01

    def test_weights_by_class_sum_100(self, portfolio):
        df = portfolio.weights_by_group("asset_class")
        assert abs(df["Weight (%)"].sum() - 100) < 0.01

    def test_weights_sorted_descending(self, portfolio):
        df = portfolio.weights_by_asset()
        vals = df["Weight (%)"].tolist()
        assert vals == sorted(vals, reverse=True)


# ── Risk Metrics ──────────────────────────────────────────────────────────────

class TestRiskMetrics:
    def test_risk_metrics_keys(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            m = portfolio.risk_metrics("AAPL")
        expected_keys = {"ticker", "ann_return_pct", "ann_volatility_pct", "sharpe_ratio",
                         "sortino_ratio", "max_drawdown_pct", "calmar_ratio",
                         "var_95_hist", "var_99_hist", "var_95_param",
                         "beta", "alpha", "treynor_ratio",
                         "best_month_pct", "best_month_date", "worst_month_pct", "worst_month_date"}
        assert expected_keys.issubset(set(m.keys()))

    def test_volatility_positive(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            m = portfolio.risk_metrics("AAPL")
        assert m["ann_volatility_pct"] > 0

    def test_max_drawdown_negative(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            m = portfolio.risk_metrics("AAPL")
        assert m["max_drawdown_pct"] <= 0

    def test_all_risk_metrics_length(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            ml = portfolio.all_risk_metrics()
        assert len(ml) == len(portfolio.assets)

    def test_portfolio_risk_returns_dict(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            pm = portfolio.portfolio_risk_metrics()
        assert isinstance(pm, dict)
        assert "ann_return_pct" in pm
        assert "sharpe_ratio" in pm


# ── Monte Carlo ───────────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_monte_carlo_runs(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            result = portfolio.monte_carlo(years=5, n_paths=10_000, seed=42)
        assert "final_median" in result
        assert result["final_median"] > 0

    def test_percentiles_ordered(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            result = portfolio.monte_carlo(years=5, n_paths=10_000, seed=42)
        assert result["final_p5"] <= result["final_median"] <= result["final_p95"]

    def test_prob_profit_range(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            result = portfolio.monte_carlo(years=5, n_paths=10_000, seed=42)
        assert 0 <= result["prob_profit"] <= 100

    def test_student_t_runs(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            result = portfolio.monte_carlo(years=3, n_paths=10_000, seed=42, distribution="t")
        assert result["final_median"] > 0


# ── Stress Testing ────────────────────────────────────────────────────────────

class TestStressTest:
    def test_stress_test_returns_list(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            results = portfolio.stress_test()
        assert isinstance(results, list)

    def test_stress_test_has_crisis_fields(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            results = portfolio.stress_test()
        for r in results:
            assert "crisis" in r
            assert "portfolio_drawdown_pct" in r


# ── Concentration ─────────────────────────────────────────────────────────────

class TestConcentration:
    def test_concentration_returns_list(self, portfolio):
        alerts = portfolio.concentration_alerts()
        assert isinstance(alerts, list)

    def test_concentration_fires_on_low_threshold(self, portfolio):
        alerts = portfolio.concentration_alerts(max_sector_pct=5.0, max_asset_pct=5.0)
        assert len(alerts) > 0

    def test_no_alerts_on_high_threshold(self, portfolio):
        alerts = portfolio.concentration_alerts(max_sector_pct=99.0, max_asset_pct=99.0)
        assert len(alerts) == 0


# ── Efficient Frontier ────────────────────────────────────────────────────────

class TestEfficientFrontier:
    def test_frontier_returns_dict(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            ef = portfolio.efficient_frontier(n_portfolios=100)
        assert isinstance(ef, dict)
        assert "frontier_returns" in ef
        assert "best_sharpe" in ef

    def test_frontier_has_optimal_weights(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            ef = portfolio.efficient_frontier(n_portfolios=100)
        assert len(ef.get("optimal_weights", {})) > 0


# ── Correlation ───────────────────────────────────────────────────────────────

class TestCorrelation:
    def test_correlation_square(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            corr = portfolio.correlation_matrix()
        assert corr.shape[0] == corr.shape[1]

    def test_correlation_diagonal_ones(self, portfolio):
        with patch("yfinance.download", side_effect=_mock_download):
            corr = portfolio.correlation_matrix()
        for i in range(len(corr)):
            assert abs(corr.iloc[i, i] - 1.0) < 0.001
