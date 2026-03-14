"""
Model Layer — a.s.r. Vermogensbeheer Portfolio Assistant
Stores asset data, performs all financial calculations including
risk metrics, stress testing, Monte Carlo (correlated GBM), and efficient frontier.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

RISK_FREE_RATE = float(__import__("os").environ.get("RISK_FREE_RATE", "0.04"))
TRADING_DAYS = 252
BENCHMARK = "SPY"

CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "2008 Financial Crisis": ("2008-09-01", "2009-03-31"),
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate Hikes": ("2022-01-03", "2022-10-12"),
    "Dot-com Bust": ("2000-03-10", "2002-10-09"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Asset:
    """Single portfolio holding."""
    ticker: str
    name: str
    sector: str
    asset_class: str
    quantity: float
    purchase_price: float
    current_price: Optional[float] = None

    @property
    def transaction_value(self) -> float:
        return self.quantity * self.purchase_price

    @property
    def current_value(self) -> float:
        return self.quantity * (self.current_price or self.purchase_price)

    @property
    def unrealised_pnl(self) -> float:
        return self.current_value - self.transaction_value

    @property
    def unrealised_pnl_pct(self) -> float:
        tv = self.transaction_value
        return (self.unrealised_pnl / tv * 100) if tv else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Portfolio:
    """Portfolio of assets with full financial analytics."""
    assets: list[Asset] = field(default_factory=list)
    _history_cache: dict[str, pd.Series] = field(default_factory=dict, repr=False)

    # ── Loading ───────────────────────────────────────────────────────────
    @classmethod
    def from_csv(cls, path: Path | str) -> Portfolio:
        """Load portfolio from CSV file."""
        df = pd.read_csv(path)
        required = {"ticker", "sector", "asset_class", "quantity", "purchase_price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        assets = [
            Asset(
                ticker=str(row["ticker"]).upper().strip(),
                name=str(row.get("name", row["ticker"])),
                sector=str(row["sector"]),
                asset_class=str(row["asset_class"]),
                quantity=float(row["quantity"]),
                purchase_price=float(row["purchase_price"]),
            )
            for _, row in df.iterrows()
        ]
        p = cls(assets=assets)
        p._fetch_live_prices()
        return p

    @classmethod
    def from_json(cls, path: Path | str) -> Portfolio:
        """Load portfolio from JSON file."""
        data = json.loads(Path(path).read_text())
        items = data if isinstance(data, list) else data.get("assets", data.get("portfolio", []))
        assets = [
            Asset(
                ticker=str(d["ticker"]).upper().strip(),
                name=str(d.get("name", d["ticker"])),
                sector=str(d["sector"]),
                asset_class=str(d["asset_class"]),
                quantity=float(d["quantity"]),
                purchase_price=float(d["purchase_price"]),
            )
            for d in items
        ]
        p = cls(assets=assets)
        p._fetch_live_prices()
        return p

    def _fetch_live_prices(self) -> None:
        """Fetch latest prices from yfinance."""
        tickers = [a.ticker for a in self.assets]
        try:
            data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
            close = data["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            for asset in self.assets:
                if asset.ticker in close.columns:
                    price = close[asset.ticker].dropna().iloc[-1] if not close[asset.ticker].dropna().empty else None
                    if price is not None and not np.isnan(float(price)):
                        asset.current_price = float(price)
        except Exception as e:
            log.warning("Failed to fetch live prices: %s", e)

    # ── Portfolio-level properties ────────────────────────────────────────
    @property
    def total_cost(self) -> float:
        return sum(a.transaction_value for a in self.assets)

    @property
    def total_value(self) -> float:
        return sum(a.current_value for a in self.assets)

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.total_cost

    @property
    def total_pnl_pct(self) -> float:
        tc = self.total_cost
        return (self.total_pnl / tc * 100) if tc else 0.0

    # ── Weights ───────────────────────────────────────────────────────────
    def weights_by_asset(self) -> pd.DataFrame:
        """Weight of each asset in the portfolio."""
        tv = self.total_value or 1
        return pd.DataFrame([
            {"Ticker": a.ticker, "Name": a.name,
             "Current Value": a.current_value, "Weight (%)": a.current_value / tv * 100}
            for a in self.assets
        ]).sort_values("Weight (%)", ascending=False)

    def weights_by_group(self, group: str) -> pd.DataFrame:
        """Aggregate weights by sector or asset_class."""
        attr = "sector" if group == "sector" else "asset_class"
        tv = self.total_value or 1
        agg: dict[str, float] = {}
        for a in self.assets:
            k = getattr(a, attr)
            agg[k] = agg.get(k, 0) + a.current_value
        return pd.DataFrame([
            {"Group": k, "Current Value": v, "Weight (%)": v / tv * 100}
            for k, v in sorted(agg.items(), key=lambda x: -x[1])
        ])

    # ── Historical data ───────────────────────────────────────────────────
    def fetch_history(self, ticker: str, period: str = "5y") -> pd.Series:
        """Fetch and cache closing price history."""
        key = f"{ticker}_{period}"
        if key not in self._history_cache:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                raise ValueError(f"No price data for {ticker}")
            close = data["Close"].squeeze() if isinstance(data.columns, pd.MultiIndex) else data["Close"]
            if isinstance(close, (int, float, np.floating)):
                raise ValueError(f"Insufficient data for {ticker}")
            close = close.dropna()
            if len(close) < 2:
                raise ValueError(f"Not enough data for {ticker}")
            self._history_cache[key] = close
        return self._history_cache[key]

    def _get_benchmark_daily_returns(self) -> pd.Series:
        """Get SPY daily returns."""
        spy = self.fetch_history(BENCHMARK)
        return spy.pct_change().dropna()

    def monthly_returns(self, ticker: str) -> pd.Series:
        return self.fetch_history(ticker).resample("ME").last().pct_change().dropna() * 100

    def quarterly_returns(self, ticker: str) -> pd.Series:
        return self.fetch_history(ticker).resample("QE").last().pct_change().dropna() * 100

    def yearly_returns(self, ticker: str) -> pd.Series:
        return self.fetch_history(ticker).resample("YE").last().pct_change().dropna() * 100

    def returns_summary(self, ticker: str) -> dict:
        return {
            "monthly": self.monthly_returns(ticker),
            "quarterly": self.quarterly_returns(ticker),
            "yearly": self.yearly_returns(ticker),
        }

    # ── Per-Asset Risk Metrics ────────────────────────────────────────────
    def risk_metrics(self, ticker: str) -> dict:
        """Full risk metrics for a single asset including VaR, Beta, Alpha, Sortino, Treynor, Calmar."""
        close = self.fetch_history(ticker)
        daily = close.pct_change().dropna()

        # Annualised via CAGR
        total_return = close.iloc[-1] / close.iloc[0]
        years = len(close) / TRADING_DAYS
        cagr = float((total_return ** (1 / years) - 1) * 100) if years > 0 else 0.0
        ann_vol = float(daily.std() * np.sqrt(TRADING_DAYS) * 100)
        sharpe = (cagr / 100 - RISK_FREE_RATE) / (ann_vol / 100) if ann_vol > 0 else 0.0

        # Sortino (downside deviation)
        downside = daily[daily < 0]
        downside_std = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else ann_vol / 100
        sortino = (cagr / 100 - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0.0

        # Max drawdown
        cum = (1 + daily).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min() * 100)

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # VaR
        var_95_hist = float(np.percentile(daily, 5) * 100)
        var_99_hist = float(np.percentile(daily, 1) * 100)
        var_95_param = float((daily.mean() - 1.645 * daily.std()) * 100)

        # Beta & Alpha vs SPY
        beta, alpha, treynor = 0.0, 0.0, 0.0
        try:
            spy_daily = self._get_benchmark_daily_returns()
            aligned = pd.concat([daily, spy_daily], axis=1).dropna()
            if len(aligned) > 10:
                aligned.columns = ["asset", "spy"]
                cov_ab = aligned["asset"].cov(aligned["spy"])
                var_b = aligned["spy"].var()
                beta = float(cov_ab / var_b) if var_b > 0 else 0.0
                spy_ann = float(aligned["spy"].mean() * TRADING_DAYS * 100)
                alpha = cagr - (RISK_FREE_RATE * 100 + beta * (spy_ann - RISK_FREE_RATE * 100))
                treynor = (cagr / 100 - RISK_FREE_RATE) / beta if beta != 0 else 0.0
        except Exception:
            pass

        # Best/worst month
        monthly = self.monthly_returns(ticker)
        if monthly.empty:
            best_m, worst_m, best_d, worst_d = 0.0, 0.0, "N/A", "N/A"
        else:
            best_m, worst_m = float(monthly.max()), float(monthly.min())
            best_d, worst_d = str(monthly.idxmax())[:7], str(monthly.idxmin())[:7]

        return {
            "ticker": ticker,
            "ann_return_pct": round(cagr, 2),
            "ann_volatility_pct": round(ann_vol, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "var_95_hist": round(var_95_hist, 2),
            "var_99_hist": round(var_99_hist, 2),
            "var_95_param": round(var_95_param, 2),
            "beta": round(beta, 3),
            "alpha": round(alpha, 2),
            "treynor_ratio": round(treynor, 3),
            "best_month_pct": round(best_m, 2),
            "best_month_date": best_d,
            "worst_month_pct": round(worst_m, 2),
            "worst_month_date": worst_d,
        }

    def all_risk_metrics(self) -> list[dict]:
        """Risk metrics for all assets, gracefully handling failures."""
        results = []
        for a in self.assets:
            try:
                results.append(self.risk_metrics(a.ticker))
            except Exception as e:
                log.warning("Risk metrics failed for %s: %s", a.ticker, e)
                results.append({"ticker": a.ticker, "ann_return_pct": 0, "ann_volatility_pct": 0,
                    "sharpe_ratio": 0, "sortino_ratio": 0, "max_drawdown_pct": 0, "calmar_ratio": 0,
                    "var_95_hist": 0, "var_99_hist": 0, "var_95_param": 0, "beta": 0, "alpha": 0,
                    "treynor_ratio": 0, "best_month_pct": 0, "best_month_date": "N/A",
                    "worst_month_pct": 0, "worst_month_date": "N/A"})
        return results

    # ── Portfolio-Level Risk ──────────────────────────────────────────────
    def _portfolio_daily_returns(self) -> tuple[pd.Series, np.ndarray, list[str]]:
        """Compute weighted portfolio daily returns. Returns (series, weights, tickers)."""
        tv = self.total_value or 1
        rets, weights, tickers = [], [], []
        for a in self.assets:
            try:
                close = self.fetch_history(a.ticker)
                rets.append(close.pct_change().dropna())
                weights.append(a.current_value / tv)
                tickers.append(a.ticker)
            except Exception:
                continue
        if not rets:
            return pd.Series(dtype=float), np.array([]), []
        combined = pd.concat(rets, axis=1).dropna()  # no fillna(0)!
        combined.columns = tickers
        w = np.array(weights[:len(tickers)])
        w /= w.sum()
        port_daily = combined.dot(w)
        return port_daily, w, tickers

    def portfolio_risk_metrics(self) -> dict:
        """Portfolio-level risk metrics including comparison vs SPY."""
        port_daily, w, tickers = self._portfolio_daily_returns()
        if port_daily.empty:
            return {}

        # CAGR
        cum = (1 + port_daily).cumprod()
        total_ret = cum.iloc[-1]
        years = len(port_daily) / TRADING_DAYS
        cagr = float((total_ret ** (1 / years) - 1) * 100) if years > 0 else 0.0
        ann_vol = float(port_daily.std() * np.sqrt(TRADING_DAYS) * 100)
        sharpe = (cagr / 100 - RISK_FREE_RATE) / (ann_vol / 100) if ann_vol > 0 else 0.0

        # Sortino
        ds = port_daily[port_daily < 0]
        ds_std = float(ds.std() * np.sqrt(TRADING_DAYS)) if len(ds) > 1 else ann_vol / 100
        sortino = (cagr / 100 - RISK_FREE_RATE) / ds_std if ds_std > 0 else 0.0

        # Max drawdown
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min() * 100)

        # VaR
        var_95 = float(np.percentile(port_daily, 5) * 100)

        # Diversification ratio
        indiv_vols = []
        for tk in tickers:
            try:
                c = self.fetch_history(tk)
                indiv_vols.append(float(c.pct_change().dropna().std() * np.sqrt(TRADING_DAYS)))
            except Exception:
                indiv_vols.append(0)
        wt = np.array(w[:len(indiv_vols)])
        weighted_avg_vol = float(np.dot(wt, indiv_vols))
        port_vol_dec = ann_vol / 100
        div_ratio = weighted_avg_vol / port_vol_dec if port_vol_dec > 0 else 1.0

        result = {
            "ann_return_pct": round(cagr, 2),
            "ann_volatility_pct": round(ann_vol, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "var_95_pct": round(var_95, 2),
            "diversification_ratio": round(div_ratio, 3),
        }

        # SPY benchmark
        try:
            spy = self.fetch_history(BENCHMARK)
            spy_d = spy.pct_change().dropna()
            aligned = pd.concat([port_daily, spy_d], axis=1).dropna()
            if len(aligned) > 10:
                aligned.columns = ["port", "spy"]
                spy_cum = (1 + aligned["spy"]).cumprod()
                spy_years = len(aligned) / TRADING_DAYS
                spy_cagr = float((spy_cum.iloc[-1] ** (1 / spy_years) - 1) * 100)
                spy_vol = float(aligned["spy"].std() * np.sqrt(TRADING_DAYS) * 100)
                spy_sharpe = (spy_cagr / 100 - RISK_FREE_RATE) / (spy_vol / 100) if spy_vol > 0 else 0.0
                spy_dd = (spy_cum - spy_cum.cummax()) / spy_cum.cummax()

                # Tracking error & information ratio
                track_err = float((aligned["port"] - aligned["spy"]).std() * np.sqrt(TRADING_DAYS) * 100)
                info_ratio = (cagr - spy_cagr) / track_err if track_err > 0 else 0.0

                # Portfolio beta
                cov_pb = aligned["port"].cov(aligned["spy"])
                var_spy = aligned["spy"].var()
                port_beta = float(cov_pb / var_spy) if var_spy > 0 else 0.0

                result.update({
                    "spy_ann_return_pct": round(spy_cagr, 2),
                    "spy_ann_volatility_pct": round(spy_vol, 2),
                    "spy_sharpe_ratio": round(spy_sharpe, 3),
                    "spy_max_drawdown_pct": round(float(spy_dd.min() * 100), 2),
                    "tracking_error_pct": round(track_err, 2),
                    "information_ratio": round(info_ratio, 3),
                    "portfolio_beta": round(port_beta, 3),
                })
        except Exception as e:
            log.warning("SPY benchmark failed: %s", e)

        return result

    # ── Correlation Matrix ────────────────────────────────────────────────
    def correlation_matrix(self) -> pd.DataFrame:
        """Pearson correlation of daily returns across all assets."""
        rets = {}
        for a in self.assets:
            try:
                rets[a.ticker] = self.fetch_history(a.ticker).pct_change().dropna()
            except Exception:
                continue
        if not rets:
            return pd.DataFrame()
        combined = pd.concat(rets.values(), axis=1).dropna()
        combined.columns = list(rets.keys())
        return combined.corr()

    # ── Stress Testing ────────────────────────────────────────────────────
    def stress_test(self) -> list[dict]:
        """Portfolio drawdown during historical crisis periods."""
        port_daily, w, tickers = self._portfolio_daily_returns()
        if port_daily.empty:
            return []

        cum = (1 + port_daily).cumprod()
        results = []
        for name, (start, end) in CRISIS_PERIODS.items():
            try:
                period = cum.loc[start:end]
                if len(period) < 2:
                    continue
                peak = period.iloc[0]
                trough = period.min()
                drawdown = float((trough / peak - 1) * 100)

                # Per-asset drawdowns
                asset_dds = {}
                for tk in tickers:
                    try:
                        c = self.fetch_history(tk)
                        ac = c.loc[start:end]
                        if len(ac) >= 2:
                            asset_dds[tk] = round(float((ac.min() / ac.iloc[0] - 1) * 100), 2)
                    except Exception:
                        pass

                results.append({
                    "crisis": name,
                    "start": start,
                    "end": end,
                    "portfolio_drawdown_pct": round(drawdown, 2),
                    "asset_drawdowns": asset_dds,
                })
            except Exception as e:
                log.warning("Stress test failed for %s: %s", name, e)
        return results

    # ── Concentration Alerts ──────────────────────────────────────────────
    def concentration_alerts(self, max_sector_pct: float = 35.0, max_asset_pct: float = 25.0) -> list[dict]:
        """Flag sectors or assets exceeding concentration limits."""
        alerts = []
        tv = self.total_value or 1

        # Per-asset check
        for a in self.assets:
            w = a.current_value / tv * 100
            if w > max_asset_pct:
                alerts.append({"type": "asset", "name": a.ticker, "weight_pct": round(w, 1),
                               "limit_pct": max_asset_pct, "excess_pct": round(w - max_asset_pct, 1)})

        # Per-sector check
        sector_w = {}
        for a in self.assets:
            sector_w[a.sector] = sector_w.get(a.sector, 0) + a.current_value / tv * 100
        for s, w in sector_w.items():
            if w > max_sector_pct:
                alerts.append({"type": "sector", "name": s, "weight_pct": round(w, 1),
                               "limit_pct": max_sector_pct, "excess_pct": round(w - max_sector_pct, 1)})
        return alerts

    # ── Rebalancing Suggestions ───────────────────────────────────────────
    def rebalancing_suggestions(self, target_sector_pct: dict | None = None,
                                 max_sector_pct: float = 30.0) -> list[dict]:
        """Suggest trades to rebalance portfolio toward target weights."""
        tv = self.total_value or 1
        suggestions = []

        # If no target provided, use equal-sector weighting capped at max
        if not target_sector_pct:
            sectors = set(a.sector for a in self.assets)
            equal_w = min(100 / len(sectors), max_sector_pct)
            target_sector_pct = {s: equal_w for s in sectors}

        # Current sector weights
        current_sector = {}
        for a in self.assets:
            current_sector[a.sector] = current_sector.get(a.sector, 0) + a.current_value

        for sector, target_pct in target_sector_pct.items():
            current_val = current_sector.get(sector, 0)
            current_pct = current_val / tv * 100
            target_val = tv * target_pct / 100
            diff_val = target_val - current_val
            diff_pct = target_pct - current_pct

            if abs(diff_pct) < 1.0:  # skip trivial changes
                continue

            # Find assets in this sector
            sector_assets = [a for a in self.assets if a.sector == sector]
            if not sector_assets:
                continue

            action = "BUY" if diff_val > 0 else "SELL"
            # Distribute change across sector assets proportionally
            for a in sector_assets:
                asset_share = a.current_value / current_val if current_val > 0 else 1 / len(sector_assets)
                asset_change = diff_val * asset_share
                shares = int(abs(asset_change) / (a.current_price or a.purchase_price))
                if shares > 0:
                    suggestions.append({
                        "action": action,
                        "ticker": a.ticker,
                        "shares": shares,
                        "value": round(abs(asset_change), 2),
                        "sector": sector,
                        "current_sector_pct": round(current_pct, 1),
                        "target_sector_pct": round(target_pct, 1),
                    })

        return suggestions

    # ── Rolling Metrics ───────────────────────────────────────────────────
    def rolling_metrics(self, window: int = TRADING_DAYS) -> dict:
        """Rolling Sharpe, volatility, beta for the portfolio."""
        port_daily, _, _ = self._portfolio_daily_returns()
        if port_daily.empty or len(port_daily) < window:
            return {}

        rolling_vol = port_daily.rolling(window).std() * np.sqrt(TRADING_DAYS)
        rolling_mean = port_daily.rolling(window).mean() * TRADING_DAYS
        rolling_sharpe = (rolling_mean - RISK_FREE_RATE) / rolling_vol

        result = {
            "dates": [str(d)[:10] for d in rolling_vol.dropna().index],
            "volatility": [round(float(v) * 100, 2) for v in rolling_vol.dropna()],
            "sharpe": [round(float(s), 3) for s in rolling_sharpe.dropna()],
        }

        # Rolling beta
        try:
            spy_d = self._get_benchmark_daily_returns()
            aligned = pd.concat([port_daily, spy_d], axis=1).dropna()
            if len(aligned) > window:
                aligned.columns = ["port", "spy"]
                rolling_cov = aligned["port"].rolling(window).cov(aligned["spy"])
                rolling_var = aligned["spy"].rolling(window).var()
                rolling_beta = (rolling_cov / rolling_var).dropna()
                result["beta"] = [round(float(b), 3) for b in rolling_beta]
                result["beta_dates"] = [str(d)[:10] for d in rolling_beta.index]
        except Exception:
            pass

        return result

    # ── Efficient Frontier ────────────────────────────────────────────────
    def efficient_frontier(self, n_portfolios: int = 3000) -> dict:
        """Monte Carlo efficient frontier with current portfolio position."""
        rets = {}
        for a in self.assets:
            try:
                rets[a.ticker] = self.fetch_history(a.ticker).pct_change().dropna()
            except Exception:
                continue
        if len(rets) < 2:
            return {}

        combined = pd.concat(rets.values(), axis=1).dropna()
        combined.columns = list(rets.keys())
        mu = combined.mean().values * TRADING_DAYS
        cov = combined.cov().values * TRADING_DAYS
        n = len(mu)
        rng = np.random.default_rng()

        frontier_ret, frontier_vol, frontier_sharpe = [], [], []
        best_sharpe_w = None
        best_sharpe = -999

        for _ in range(n_portfolios):
            w = rng.random(n)
            w /= w.sum()
            ret = float(np.dot(w, mu) * 100)
            vol = float(np.sqrt(np.dot(w, np.dot(cov, w))) * 100)
            sr = (ret / 100 - RISK_FREE_RATE) / (vol / 100) if vol > 0 else 0
            frontier_ret.append(round(ret, 2))
            frontier_vol.append(round(vol, 2))
            frontier_sharpe.append(round(sr, 3))
            if sr > best_sharpe:
                best_sharpe = sr
                best_sharpe_w = w.copy()

        # Current portfolio position
        tv = self.total_value or 1
        cur_w = np.array([a.current_value / tv for a in self.assets if a.ticker in combined.columns])
        cur_w = cur_w[:n]
        if len(cur_w) == n:
            cur_w /= cur_w.sum()
            cur_ret = float(np.dot(cur_w, mu) * 100)
            cur_vol = float(np.sqrt(np.dot(cur_w, np.dot(cov, cur_w))) * 100)
        else:
            cur_ret, cur_vol = 0, 0

        return {
            "frontier_returns": frontier_ret,
            "frontier_vols": frontier_vol,
            "frontier_sharpes": frontier_sharpe,
            "current_return": round(cur_ret, 2),
            "current_vol": round(cur_vol, 2),
            "optimal_weights": {list(rets.keys())[i]: round(float(best_sharpe_w[i]) * 100, 1) for i in range(n)} if best_sharpe_w is not None else {},
            "current_weights": {list(rets.keys())[i]: round(float(cur_w[i]) * 100, 1) for i in range(len(cur_w))} if len(cur_w) == n else {},
            "best_sharpe": round(best_sharpe, 3),
        }

    # ── Monte Carlo (Correlated Multi-Asset GBM) ─────────────────────────
    def monte_carlo(self, years: int = 15, n_paths: int = 100_000,
                    seed: Optional[int] = None, distribution: str = "normal") -> dict:
        """Correlated multi-asset GBM simulation via Cholesky decomposition."""
        tv = self.total_value or 1
        rets, weights, tickers = [], [], []
        for a in self.assets:
            try:
                close = self.fetch_history(a.ticker)
                rets.append(close.pct_change().dropna())
                weights.append(a.current_value / tv)
                tickers.append(a.ticker)
            except Exception:
                continue
        if not rets:
            return {}

        combined = pd.concat(rets, axis=1).dropna()
        combined.columns = tickers
        n_assets = len(tickers)
        w = np.array(weights[:n_assets])
        w /= w.sum()

        mu_vec = combined.mean().values  # per-asset daily drift
        cov_matrix = combined.cov().values
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback: add small diagonal for positive definiteness
            L = np.linalg.cholesky(cov_matrix + np.eye(n_assets) * 1e-8)

        trading_days = years * TRADING_DAYS
        rng = np.random.default_rng(seed)
        batch = 10_000
        n_batches = n_paths // batch

        year_ends = [min(y * TRADING_DAYS - 1, trading_days - 1) for y in range(1, years + 1)]
        all_finals = []
        year_snapshots = [[] for _ in range(years)]

        for _ in range(n_batches):
            if distribution == "t":
                Z = rng.standard_t(df=5, size=(n_assets, trading_days, batch))
            else:
                Z = rng.standard_normal((n_assets, trading_days, batch))

            # Correlate: (n_assets, trading_days, batch)
            corr_Z = np.tensordot(L, Z, axes=(1, 0))

            # Simulate each asset: log returns using correlated noise
            # corr_Z already has the right covariance structure from Cholesky
            # So we use: log_r_i = (mu_i - 0.5 * var_i) + corr_Z_i
            # where corr_Z_i already contains the volatility scaling
            log_r = np.zeros_like(corr_Z)
            for i in range(n_assets):
                daily_var = cov_matrix[i, i]
                log_r[i] = (mu_vec[i] - 0.5 * daily_var) + corr_Z[i]

            # Cumulative paths per asset: shape (n_assets, trading_days, batch)
            cum_log = np.cumsum(log_r, axis=1)
            # Price paths relative to initial value per asset
            # Portfolio value at each step = sum(w_i * initial_value_i * exp(cum_log_i))
            asset_values = np.zeros((n_assets,))
            for i, a_tk in enumerate(tickers):
                for a in self.assets:
                    if a.ticker == a_tk:
                        asset_values[i] = a.current_value
                        break

            # Portfolio value at each time step
            port_paths = np.zeros((trading_days, batch))
            for i in range(n_assets):
                port_paths += asset_values[i] * np.exp(cum_log[i])

            all_finals.extend(port_paths[-1].tolist())
            for yi, idx in enumerate(year_ends):
                year_snapshots[yi].extend(port_paths[idx].tolist())

        final_arr = np.array(all_finals)
        percentiles = {}
        for pv in [5, 25, 50, 75, 95]:
            percentiles[str(pv)] = [round(float(np.percentile(year_snapshots[yi], pv)), 2) for yi in range(years)]

        var_5 = float(np.percentile(final_arr, 5))
        cvar_5 = float(final_arr[final_arr <= var_5].mean()) if np.any(final_arr <= var_5) else var_5

        return {
            "S0": round(tv, 2),
            "years": years,
            "n_paths": n_paths,
            "distribution": distribution,
            "mu_annual": [round(float(m * TRADING_DAYS), 4) for m in mu_vec],
            "year_labels": list(range(1, years + 1)),
            "percentiles": percentiles,
            "final_median": round(float(np.median(final_arr)), 2),
            "final_p5": round(float(np.percentile(final_arr, 5)), 2),
            "final_p95": round(float(np.percentile(final_arr, 95)), 2),
            "prob_profit": round(float(np.mean(final_arr > tv) * 100), 1),
            "cvar_5": round(cvar_5, 2),
        }