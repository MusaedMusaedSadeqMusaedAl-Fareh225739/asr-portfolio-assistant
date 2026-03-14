"""
Controller Layer — a.s.r. Vermogensbeheer Portfolio Assistant
OpenAI tool-use agent with proper agentic loop. 10 tools.
"""
from __future__ import annotations

import json
import logging
import os
import re

from openai import OpenAI
from models.portfolio import Portfolio
from views import display as view

log = logging.getLogger(__name__)


def _clean(text: str) -> str:
    """Strip Rich markup tags so tool results are plain text for the LLM."""
    return re.sub(r"\[/?[a-zA-Z0-9_ ]+\]", "", str(text))


SYSTEM_PROMPT = """You are a senior quantitative portfolio analyst at a.s.r. Vermogensbeheer, a major Dutch institutional asset manager.

You have access to tools that compute real financial data from live market prices via yfinance. ALWAYS use tools to answer questions — never invent or estimate numbers.

RESPONSE FORMAT RULES (CRITICAL):
- The UI automatically displays detailed tables and charts from tool results. DO NOT repeat every row of data in your text response.
- Instead, write a SHORT summary (2-4 sentences) highlighting key insights: totals, best/worst performers, notable patterns.
- Example GOOD response: "Your portfolio is worth EUR 75,452 with an unrealised P&L of EUR -44,798 (-37.25%). Top performers are GLD (+171%) and JPM (+118%). AMZN and GOOGL are significant drags at -93.5% and -89.2% respectively."
- Example BAD response: listing every single asset with all its numbers — the table already shows this.
- For risk metrics: highlight the best/worst Sharpe, any concerning drawdowns, and the portfolio-level summary.
- For weights: note the biggest concentration and any imbalances.
- Keep it professional and concise. The data table speaks for itself.

TOOL USAGE RULES:
- For portfolio overview: use show_portfolio_overview
- For weights: use show_weights with by="all" for complete breakdown
- For returns: use show_historical_returns with tickers=["ALL"] for all assets
- For risk metrics: use show_risk_metrics with level="both" for per-asset + portfolio
- For correlation: use show_correlation
- For Monte Carlo: use run_monte_carlo
- For stress testing: use run_stress_test
- For efficient frontier: use show_efficient_frontier
- For concentration risk: use show_concentration_risk
- For rolling metrics: use show_rolling_metrics

You can chain multiple tools in one turn. For example, if asked "analyze my portfolio risk", you might call show_risk_metrics, then run_stress_test, then show_concentration_risk.

If a tool returns an error, tell the user honestly — never fabricate data. Use EUR for currency."""

TOOLS = [
    {"type": "function", "function": {
        "name": "show_portfolio_overview",
        "description": "Show all assets with name, sector, class, qty, prices, values, P&L.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "show_weights",
        "description": "Show portfolio weights by asset, sector, or asset class.",
        "parameters": {"type": "object", "properties": {
            "by": {"type": "string", "enum": ["asset", "sector", "asset_class", "all"]},
            "chart": {"type": "boolean", "default": False}},
            "required": ["by"]}}},
    {"type": "function", "function": {
        "name": "show_historical_returns",
        "description": "Fetch real historical monthly/quarterly/yearly returns. Use ['ALL'] for all assets.",
        "parameters": {"type": "object", "properties": {
            "tickers": {"type": "array", "items": {"type": "string"}},
            "chart": {"type": "boolean", "default": False}},
            "required": ["tickers"]}}},
    {"type": "function", "function": {
        "name": "show_risk_metrics",
        "description": "Show Sharpe, Sortino, Beta, Alpha, VaR, max drawdown per asset and/or portfolio vs SPY.",
        "parameters": {"type": "object", "properties": {
            "level": {"type": "string", "enum": ["asset", "portfolio", "both"]},
            "chart": {"type": "boolean", "default": False}},
            "required": ["level"]}}},
    {"type": "function", "function": {
        "name": "show_correlation",
        "description": "Show correlation matrix of daily returns for diversification analysis.",
        "parameters": {"type": "object", "properties": {
            "chart": {"type": "boolean", "default": True}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "run_monte_carlo",
        "description": "Run correlated multi-asset GBM Monte Carlo simulation over 15 years with 100K paths.",
        "parameters": {"type": "object", "properties": {
            "distribution": {"type": "string", "enum": ["normal", "t"], "default": "normal"},
            "chart": {"type": "boolean", "default": True}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "run_stress_test",
        "description": "Show portfolio drawdowns during historical crises (2008, COVID, 2022 Rate Hikes, Dot-com).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "show_efficient_frontier",
        "description": "Show Markowitz efficient frontier with current portfolio position and optimal weights.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "show_concentration_risk",
        "description": "Check if any sector exceeds 35% or any single asset exceeds 25% of portfolio.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "show_rolling_metrics",
        "description": "Show 12-month rolling Sharpe ratio, volatility, and beta for the portfolio.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


class PortfolioController:
    """GPT-4o agent with proper agentic loop."""

    def __init__(self, portfolio: Portfolio, api_key: str | None = None,
                 model: str | None = None) -> None:
        self.portfolio = portfolio
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.conversation_history: list[dict] = []

    def _execute_tool(self, name: str, inputs: dict) -> str:
        """Execute a tool and return structured data as text for GPT."""
        p = self.portfolio
        try:
            if name == "show_portfolio_overview":
                view.show_portfolio_overview(p.assets)
                view.show_portfolio_summary(p)
                lines = []
                for a in p.assets:
                    cp = f"EUR{a.current_price:,.2f}" if a.current_price else "N/A"
                    lines.append(
                        f"{a.ticker}: {a.name}, {a.sector}, {a.asset_class}, "
                        f"Qty={int(a.quantity)}, Buy=EUR{a.purchase_price:,.2f}, "
                        f"Current={cp}, Value=EUR{a.current_value:,.2f}, "
                        f"P&L=EUR{a.unrealised_pnl:+,.2f} ({a.unrealised_pnl_pct:+.2f}%)")
                lines.append(f"\nTotal: {len(p.assets)} assets, Cost=EUR{p.total_cost:,.2f}, "
                             f"Value=EUR{p.total_value:,.2f}, P&L=EUR{p.total_pnl:,.2f} ({p.total_pnl_pct:+.2f}%)")
                return _clean("\n".join(lines))

            elif name == "show_weights":
                by = inputs.get("by", "all")
                df_a = p.weights_by_asset()
                df_s = p.weights_by_group("sector")
                df_c = p.weights_by_group("asset_class")
                if by in ("asset", "all"):
                    view.show_weights_by_asset(df_a)
                if by in ("sector", "all"):
                    view.show_weights_by_group(df_s, "Sector")
                if by in ("asset_class", "all"):
                    view.show_weights_by_group(df_c, "Asset Class")
                lines = []
                if by in ("asset", "all"):
                    lines.append("BY ASSET:")
                    for _, r in df_a.iterrows():
                        lines.append(f"  {r['Ticker']}: EUR{r['Current Value']:,.2f} ({r['Weight (%)']:.1f}%)")
                if by in ("sector", "all"):
                    lines.append("BY SECTOR:")
                    for _, r in df_s.iterrows():
                        lines.append(f"  {r['Group']}: EUR{r['Current Value']:,.2f} ({r['Weight (%)']:.1f}%)")
                if by in ("asset_class", "all"):
                    lines.append("BY ASSET CLASS:")
                    for _, r in df_c.iterrows():
                        lines.append(f"  {r['Group']}: EUR{r['Current Value']:,.2f} ({r['Weight (%)']:.1f}%)")
                return _clean("\n".join(lines))

            elif name == "show_historical_returns":
                raw = inputs.get("tickers", ["ALL"])
                valid = {a.ticker for a in p.assets}
                tickers = list(valid) if "ALL" in [t.upper() for t in raw] else [t.upper() for t in raw if t.upper() in valid]
                if not tickers:
                    return "No valid tickers found."
                saved = []
                for ticker in tickers:
                    view.print_info(f"Fetching returns for {ticker}...")
                    ret = p.returns_summary(ticker)
                    view.show_returns_table(ret, ticker)
                    if inputs.get("chart"):
                        path = view.plot_returns(ret, ticker)
                        saved.append(path)
                parts = [f"Returns displayed for: {', '.join(tickers)}."]
                for tk in tickers:
                    yr = p.yearly_returns(tk)
                    if not yr.empty:
                        best_yr = f"{yr.max():+.1f}%"
                        worst_yr = f"{yr.min():+.1f}%"
                        parts.append(f"{tk}: Best year {best_yr}, Worst year {worst_yr}")
                return _clean("\n".join(parts))

            elif name == "show_risk_metrics":
                level = inputs.get("level", "both")
                parts = []
                if level in ("asset", "both"):
                    view.print_info("Computing per-asset risk metrics...")
                    ml = p.all_risk_metrics()
                    view.show_risk_metrics(ml)
                    for m in ml:
                        parts.append(
                            f"{m['ticker']}: Return={m['ann_return_pct']:+.2f}%, "
                            f"Vol={m['ann_volatility_pct']:.2f}%, Sharpe={m['sharpe_ratio']:.3f}, "
                            f"Sortino={m['sortino_ratio']:.3f}, Beta={m['beta']:.3f}, "
                            f"Alpha={m['alpha']:+.2f}%, MaxDD={m['max_drawdown_pct']:.2f}%, "
                            f"VaR95={m['var_95_hist']:.2f}%")
                if level in ("portfolio", "both"):
                    view.print_info("Computing portfolio risk vs SPY...")
                    pm = p.portfolio_risk_metrics()
                    view.show_portfolio_risk(pm)
                    parts.append(
                        f"\nPortfolio: Return={pm.get('ann_return_pct',0):+.2f}%, "
                        f"Vol={pm.get('ann_volatility_pct',0):.2f}%, "
                        f"Sharpe={pm.get('sharpe_ratio',0):.3f}, "
                        f"Sortino={pm.get('sortino_ratio',0):.3f}, "
                        f"MaxDD={pm.get('max_drawdown_pct',0):.2f}%, "
                        f"DivRatio={pm.get('diversification_ratio',0):.3f}")
                    if "spy_sharpe_ratio" in pm:
                        parts.append(
                            f"SPY: Return={pm['spy_ann_return_pct']:+.2f}%, "
                            f"Sharpe={pm['spy_sharpe_ratio']:.3f}, "
                            f"TrackErr={pm.get('tracking_error_pct',0):.2f}%, "
                            f"InfoRatio={pm.get('information_ratio',0):.3f}, "
                            f"PortBeta={pm.get('portfolio_beta',0):.3f}")
                return _clean("\n".join(parts))

            elif name == "show_correlation":
                view.print_info("Computing correlation matrix...")
                corr_df = p.correlation_matrix()
                if corr_df.empty:
                    return "Could not compute correlation matrix."
                if inputs.get("chart", True):
                    path = view.plot_correlation_heatmap(corr_df)
                    view.print_chart_saved(path)
                vals = corr_df.where(corr_df != 1.0).stack()
                most = vals.idxmax()
                least = vals.idxmin()
                return _clean(
                    f"Most correlated: {most[0]} & {most[1]} ({vals.max():.2f}). "
                    f"Least correlated: {least[0]} & {least[1]} ({vals.min():.2f}).")

            elif name == "run_monte_carlo":
                dist = inputs.get("distribution", "normal")
                view.print_info(f"Running 100K-path Monte Carlo ({dist})...")
                result = p.monte_carlo(years=15, n_paths=100_000, distribution=dist)
                view.show_monte_carlo_summary(result)
                if inputs.get("chart", True):
                    path = view.plot_monte_carlo(result)
                    view.print_chart_saved(path)
                return _clean(
                    f"Monte Carlo ({dist}): Start=EUR{result['S0']:,.0f}, "
                    f"Median@15yr=EUR{result['final_median']:,.0f}, "
                    f"Range(5-95th)=EUR{result['final_p5']:,.0f}-EUR{result['final_p95']:,.0f}, "
                    f"CVaR(5%)=EUR{result['cvar_5']:,.0f}, "
                    f"P(profit)={result['prob_profit']:.1f}%")

            elif name == "run_stress_test":
                view.print_info("Running stress tests...")
                results = p.stress_test()
                view.show_stress_test(results)
                lines = ["Stress test results:"]
                for r in results:
                    lines.append(f"  {r['crisis']} ({r['start']} to {r['end']}): "
                                 f"Portfolio drawdown {r['portfolio_drawdown_pct']:+.2f}%")
                    for tk, dd in r.get("asset_drawdowns", {}).items():
                        lines.append(f"    {tk}: {dd:+.2f}%")
                return _clean("\n".join(lines))

            elif name == "show_efficient_frontier":
                view.print_info("Computing efficient frontier...")
                ef = p.efficient_frontier()
                if not ef:
                    return "Could not compute efficient frontier."
                view.show_efficient_frontier(ef)
                lines = [f"Efficient Frontier: Best Sharpe={ef['best_sharpe']:.3f}",
                         f"Current portfolio: Return={ef['current_return']:.2f}%, Vol={ef['current_vol']:.2f}%",
                         "Optimal weights vs current:"]
                for tk in ef.get("optimal_weights", {}):
                    opt = ef["optimal_weights"][tk]
                    cur = ef.get("current_weights", {}).get(tk, 0)
                    lines.append(f"  {tk}: Optimal={opt:.1f}% Current={cur:.1f}% (Δ={opt-cur:+.1f}%)")
                return _clean("\n".join(lines))

            elif name == "show_concentration_risk":
                alerts = p.concentration_alerts()
                view.show_concentration_alerts(alerts)
                if not alerts:
                    return "No concentration risk alerts. All sectors < 35% and all assets < 25%."
                lines = ["Concentration alerts:"]
                for a in alerts:
                    lines.append(f"  ⚠ {a['type'].title()} '{a['name']}': {a['weight_pct']:.1f}% "
                                 f"(limit {a['limit_pct']:.0f}%, excess {a['excess_pct']:.1f}%)")
                return _clean("\n".join(lines))

            elif name == "show_rolling_metrics":
                view.print_info("Computing rolling metrics...")
                rm = p.rolling_metrics()
                if not rm:
                    return "Not enough data for rolling metrics."
                view.show_rolling_metrics(rm)
                latest_sharpe = rm["sharpe"][-1] if rm["sharpe"] else 0
                latest_vol = rm["volatility"][-1] if rm["volatility"] else 0
                return _clean(f"Rolling 12-month: Sharpe={latest_sharpe:.3f}, Vol={latest_vol:.2f}%")

            return f"Unknown tool: {name}"
        except Exception as exc:
            log.error("Tool '%s' error: %s", name, exc)
            view.print_error(f"Tool '{name}' error: {exc}")
            return f"Error executing {name}: {exc}"

    def chat(self, user_message: str) -> None:
        """Proper agentic loop — tool_choice=auto, multi-step chaining."""
        self.conversation_history.append({"role": "user", "content": user_message})

        asset_summary = ", ".join(f"{a.ticker} ({a.asset_class}, {a.sector})" for a in self.portfolio.assets)
        system_msg = {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\nPortfolio: {len(self.portfolio.assets)} assets: {asset_summary}. "
                       f"Total value: EUR{self.portfolio.total_value:,.2f}."
        }

        max_iterations = 10
        for iteration in range(max_iterations):
            messages = [system_msg] + self.conversation_history
            view.print_thinking("Thinking...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=2048,
            )
            msg = response.choices[0].message
            self.conversation_history.append(msg.model_dump())

            # No tool calls → LLM is done, output text
            if not msg.tool_calls:
                if msg.content:
                    view.print_llm_response(msg.content)
                break

            # Execute all tool calls
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_inputs = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_inputs = {}
                view.print_info(f"Tool: {tool_name}")
                result_str = self._execute_tool(tool_name, tool_inputs)
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
            # Loop continues — LLM sees results and decides next action
        else:
            log.warning("Agent hit max iterations (%d)", max_iterations)
            view.print_llm_response("I've completed the analysis. Let me know if you need anything else.")
