"""
View Layer - Portfolio Assistant (a.s.r. Vermogensbeheer)
Rich terminal tables + professional matplotlib charts.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)
_DARK, _PANEL = "#0a0d11", "#111518"


def _c(val: float) -> str:
    return "green" if val >= 0 else "red"

def _dark_ax(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_color("#444")


# ── Portfolio overview ───────────────────────────────────────────────────────

def show_portfolio_overview(assets) -> None:
    t = Table(title="📊 Portfolio Overview", box=box.ROUNDED,
              header_style="bold cyan", border_style="cyan")
    for col, just in [("Ticker","center"),("Name","left"),("Sector","left"),
                      ("Class","left"),("Qty","right"),("Buy Price","right"),
                      ("Cur Price","right"),("Txn Value","right"),
                      ("Cur Value","right"),("P&L","right"),("P&L %","right")]:
        t.add_column(col, justify=just)
    for a in assets:
        cp = f"€{a.current_price:,.2f}" if a.current_price else "N/A"
        c = _c(a.unrealised_pnl)
        s = "+" if a.unrealised_pnl >= 0 else ""
        t.add_row(f"[bold]{a.ticker}[/bold]", a.name, a.sector, a.asset_class,
                  f"{a.quantity:,.0f}", f"€{a.purchase_price:,.2f}", cp,
                  f"€{a.transaction_value:,.2f}", f"€{a.current_value:,.2f}",
                  f"[{c}]{s}€{a.unrealised_pnl:,.2f}[/{c}]",
                  f"[{c}]{s}{a.unrealised_pnl_pct:.2f}%[/{c}]")
    console.print(t)


def show_portfolio_summary(portfolio) -> None:
    c = _c(portfolio.total_pnl)
    s = "+" if portfolio.total_pnl >= 0 else ""
    text = (f"[bold]Total Cost:[/bold]          €{portfolio.total_cost:,.2f}\n"
            f"[bold]Total Current Value:[/bold] €{portfolio.total_value:,.2f}\n"
            f"[bold]Unrealised P&L:[/bold]      [{c}]{s}€{portfolio.total_pnl:,.2f}[/{c}]\n"
            f"[bold]P&L %:[/bold]               [{c}]{s}{portfolio.total_pnl_pct:.2f}%[/{c}]")
    console.print(Panel(text, title="💼 Portfolio Summary", border_style="blue", expand=False))


# ── Weights ──────────────────────────────────────────────────────────────────

def show_weights_by_asset(df: pd.DataFrame) -> None:
    t = Table(title="⚖️  Asset Weights", box=box.ROUNDED, header_style="bold magenta")
    t.add_column("Ticker", justify="center")
    t.add_column("Name")
    t.add_column("Current Value", justify="right")
    t.add_column("Weight", justify="right")
    for _, row in df.iterrows():
        bar = "█" * int(row["Weight (%)"]/5) + "░" * (20 - int(row["Weight (%)"]/5))
        t.add_row(f"[bold]{row['Ticker']}[/bold]", row["Name"],
                  f"€{row['Current Value']:,.2f}",
                  f"{row['Weight (%)']:.2f}% [dim]{bar}[/dim]")
    console.print(t)


def show_weights_by_group(df: pd.DataFrame, group_label: str) -> None:
    t = Table(title=f"⚖️  Weights by {group_label}", box=box.ROUNDED, header_style="bold magenta")
    t.add_column(group_label)
    t.add_column("Current Value", justify="right")
    t.add_column("Weight", justify="right")
    for _, row in df.iterrows():
        bar = "█" * int(row["Weight (%)"]/5) + "░" * (20 - int(row["Weight (%)"]/5))
        t.add_row(row["Group"], f"€{row['Current Value']:,.2f}",
                  f"{row['Weight (%)']:.2f}% [dim]{bar}[/dim]")
    console.print(t)


# ── Returns ──────────────────────────────────────────────────────────────────

def show_returns_table(returns_data: dict, ticker: str) -> None:
    for period, series in [("Monthly", returns_data["monthly"]),
                            ("Quarterly", returns_data["quarterly"]),
                            ("Yearly", returns_data["yearly"])]:
        t = Table(title=f"📈 {ticker} — {period} Returns",
                  box=box.SIMPLE_HEAVY, header_style="bold yellow")
        t.add_column("Period")
        t.add_column("Return", justify="right")
        t.add_column("Bar")
        for date, val in series.tail(24).items():
            label = str(date)[:7] if period == "Monthly" else str(date)[:10]
            c = "green" if val >= 0 else "red"
            bar = ("+" if val >= 0 else "-") * min(int(abs(val) / 2), 20)
            t.add_row(label, f"[{c}]{val:+.2f}%[/{c}]", f"[{c}]{bar}[/{c}]")
        console.print(t)


# ── Risk metrics ─────────────────────────────────────────────────────────────

def show_risk_metrics(metrics_list: list[dict]) -> None:
    t = Table(title="📉 Risk Metrics per Asset (5-year, annualised)",
              box=box.ROUNDED, header_style="bold red")
    for col, just in [("Ticker","center"),("Ann. Return","right"),
                      ("Volatility","right"),("Sharpe","right"),
                      ("Max Drawdown","right"),("Best Month","right"),
                      ("Worst Month","right")]:
        t.add_column(col, justify=just)
    for m in metrics_list:
        sr_col = "green" if m["sharpe_ratio"] >= 1 else "yellow" if m["sharpe_ratio"] >= 0.5 else "red"
        t.add_row(
            f"[bold]{m['ticker']}[/bold]",
            f"[{_c(m['ann_return_pct'])}]{m['ann_return_pct']:+.2f}%[/{_c(m['ann_return_pct'])}]",
            f"{m['ann_volatility_pct']:.2f}%",
            f"[{sr_col}]{m['sharpe_ratio']:.3f}[/{sr_col}]",
            f"[red]{m['max_drawdown_pct']:.2f}%[/red]",
            f"[green]+{m['best_month_pct']:.2f}% ({m['best_month_date']})[/green]",
            f"[red]{m['worst_month_pct']:.2f}% ({m['worst_month_date']})[/red]",
        )
    console.print(t)


def show_portfolio_risk(metrics: dict) -> None:
    has_spy = "spy_ann_return_pct" in metrics
    text = (
        f"[bold]Portfolio Risk Metrics (annualised, 5-year)[/bold]\n\n"
        f"  Ann. Return:     [{_c(metrics['ann_return_pct'])}]{metrics['ann_return_pct']:+.2f}%[/{_c(metrics['ann_return_pct'])}]\n"
        f"  Ann. Volatility: {metrics['ann_volatility_pct']:.2f}%\n"
        f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.3f}  [dim](risk-free: 4%)[/dim]\n"
        f"  Max Drawdown:    [red]{metrics['max_drawdown_pct']:.2f}%[/red]\n"
    )
    if has_spy:
        text += (
            f"\n[bold]vs SPY Benchmark[/bold]\n"
            f"  SPY Ann. Return:  {metrics['spy_ann_return_pct']:+.2f}%\n"
            f"  SPY Volatility:   {metrics['spy_ann_volatility_pct']:.2f}%\n"
            f"  SPY Sharpe:       {metrics['spy_sharpe_ratio']:.3f}\n"
            f"  SPY Max Drawdown: [red]{metrics['spy_max_drawdown_pct']:.2f}%[/red]\n"
        )
    console.print(Panel(text, title="📉 Portfolio Risk Analysis", border_style="red", expand=False))


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def show_monte_carlo_summary(result: dict) -> None:
    s0, med = result["S0"], result["final_median"]
    p5, p95 = result["final_p5"], result["final_p95"]
    cvar = result.get("cvar_5", p5)
    mu_list = result.get('mu_annual', [])
    avg_drift = sum(mu_list) / len(mu_list) * 100 if mu_list else 0
    text = (
        f"[bold]Simulation Parameters[/bold]\n"
        f"  Paths:             {result['n_paths']:,}\n"
        f"  Horizon:           {result['years']} years\n"
        f"  Distribution:      {result.get('distribution', 'normal')}\n"
        f"  Avg. Annual Drift: {avg_drift:.2f}%\n\n"
        f"[bold]Results at Year {result['years']}[/bold]\n"
        f"  Starting Value:    €{s0:,.2f}\n"
        f"  Median outcome:    [cyan]€{med:,.2f}[/cyan]  ({(med/s0-1)*100:+.1f}%)\n"
        f"  5th percentile:    [red]€{p5:,.2f}[/red]  ({(p5/s0-1)*100:+.1f}%)\n"
        f"  95th percentile:   [green]€{p95:,.2f}[/green]  ({(p95/s0-1)*100:+.1f}%)\n"
        f"  CVaR (5%):         [bold red]€{cvar:,.2f}[/bold red]  [dim](expected loss in worst 5%)[/dim]\n"
        f"  Prob. of profit:   [bold green]{result['prob_profit']:.1f}%[/bold green]"
    )
    console.print(Panel(text, title="🎲 Monte Carlo Simulation (100,000 paths)", border_style="magenta", expand=False))


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_returns(returns_data: dict, ticker: str) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"{ticker} — Historical Returns", fontsize=14, fontweight="bold", color="white")
    for ax, (title, series) in zip(axes, [
        ("Monthly Returns (%)", returns_data["monthly"].tail(36)),
        ("Quarterly Returns (%)", returns_data["quarterly"].tail(20)),
        ("Yearly Returns (%)", returns_data["yearly"]),
    ]):
        colours = ["#4ade80" if v >= 0 else "#f87171" for v in series.values]
        ax.bar(range(len(series)), series.values, color=colours, edgecolor="none")
        ax.axhline(0, color="white", lw=0.5, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Return (%)")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([str(d)[:7] for d in series.index], rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        _dark_ax(ax)
    fig.patch.set_facecolor(_DARK)
    plt.tight_layout()
    path = CHARTS_DIR / f"{ticker}_returns.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    return str(path)


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_PANEL)
    mask = np.zeros_like(corr_df.values)
    np.fill_diagonal(mask, 0)
    sns.heatmap(
        corr_df, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=-1, vmax=1, center=0,
        linewidths=0.5, linecolor="#333",
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Asset Correlation Matrix (Daily Returns)", color="white", fontsize=13, pad=15)
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#444")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")
    plt.tight_layout()
    path = CHARTS_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    return str(path)


def plot_risk_metrics(metrics_list: list[dict]) -> str:
    tickers = [m["ticker"] for m in metrics_list]
    vols = [m["ann_volatility_pct"] for m in metrics_list]
    returns = [m["ann_return_pct"] for m in metrics_list]
    sharpes = [m["sharpe_ratio"] for m in metrics_list]
    drawdowns = [abs(m["max_drawdown_pct"]) for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Risk Metrics Dashboard", fontsize=14, fontweight="bold", color="white")

    # Return vs Volatility scatter
    ax = axes[0][0]
    colours = ["#4ade80" if r >= 0 else "#f87171" for r in returns]
    scatter = ax.scatter(vols, returns, s=120, c=colours, zorder=5)
    for i, t in enumerate(tickers):
        ax.annotate(t, (vols[i], returns[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="white")
    ax.axhline(0, color="white", lw=0.5, alpha=0.5)
    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Return vs Risk", color="white")
    ax.grid(alpha=0.3)
    _dark_ax(ax)

    # Sharpe ratios bar chart
    ax = axes[0][1]
    s_colours = ["#4ade80" if s >= 1 else "#facc15" if s >= 0.5 else "#f87171" for s in sharpes]
    ax.barh(tickers, sharpes, color=s_colours, edgecolor="none")
    ax.axvline(1.0, color="white", lw=1, ls="--", alpha=0.6, label="Good (1.0)")
    ax.axvline(0.5, color="#facc15", lw=1, ls="--", alpha=0.6, label="Acceptable (0.5)")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratios (rf=4%)", color="white")
    ax.legend(fontsize=8, facecolor=_PANEL, labelcolor="white")
    ax.grid(axis="x", alpha=0.3)
    _dark_ax(ax)

    # Max drawdown
    ax = axes[1][0]
    ax.barh(tickers, drawdowns, color="#f87171", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_title("Maximum Drawdown", color="white")
    ax.grid(axis="x", alpha=0.3)
    _dark_ax(ax)

    # Volatility bar chart
    ax = axes[1][1]
    ax.bar(tickers, vols, color="#22c55e", edgecolor="none", alpha=0.8)
    ax.set_ylabel("Annualised Volatility (%)")
    ax.set_title("Volatility per Asset", color="white")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    _dark_ax(ax)

    fig.patch.set_facecolor(_DARK)
    plt.tight_layout()
    path = CHARTS_DIR / "risk_metrics.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    return str(path)


def plot_monte_carlo(result: dict) -> str:
    years = result["year_labels"]
    pct = result["percentiles"]
    S0 = result["S0"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Portfolio Monte Carlo — 100,000 Paths, 15 Years",
                 fontsize=13, fontweight="bold", color="white")
    ax1.fill_between(years, pct["5"], pct["95"], alpha=0.15, color="#22c55e", label="5–95th pct")
    ax1.fill_between(years, pct["25"], pct["75"], alpha=0.30, color="#22c55e", label="25–75th pct")
    ax1.plot(years, pct["50"], color="#facc15", lw=2.5, label="Median")
    ax1.plot(years, pct["5"], color="#f87171", lw=1, ls="--", label="5th pct")
    ax1.plot(years, pct["95"], color="#4ade80", lw=1, ls="--", label="95th pct")
    ax1.axhline(S0, color="white", lw=0.8, ls=":", alpha=0.6, label="Start")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1e3:.0f}k"))
    ax1.set_xlabel("Year"); ax1.set_ylabel("Portfolio Value (€)")
    ax1.set_title("Fan Chart — Path Percentiles", color="white")
    ax1.legend(fontsize=8, facecolor=_PANEL, labelcolor="white")
    ax1.grid(alpha=0.2); _dark_ax(ax1)
    # Right panel: key metrics visualization instead of histogram
    metrics = [
        ("Start", S0, "white"),
        ("5th Pct", result["final_p5"], "#f87171"),
        ("Median", result["final_median"], "#facc15"),
        ("95th Pct", result["final_p95"], "#4ade80"),
        ("CVaR 5%", result.get("cvar_5", result["final_p5"]), "#c0392b"),
    ]
    names = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    ax2.barh(names, vals, color=colors, edgecolor="none", alpha=0.85, height=0.5)
    for i, v in enumerate(vals):
        ax2.text(v + max(vals)*0.02, i, f"€{v:,.0f}", va="center", color="white", fontsize=9)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1e3:.0f}k"))
    ax2.set_xlabel("Portfolio Value (€)")
    ax2.set_title("Key Outcomes at Year 15", color="white")
    ax2.grid(axis="x", alpha=0.2); _dark_ax(ax2)
    fig.patch.set_facecolor(_DARK)
    plt.tight_layout()
    path = CHARTS_DIR / "monte_carlo.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    return str(path)


def plot_weights_pie(df_a: pd.DataFrame, df_s: pd.DataFrame, df_c: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Portfolio Weight Breakdown", fontsize=13, fontweight="bold", color="white")
    for ax, (df, col, title) in zip(axes, [
        (df_a, "Ticker", "By Asset"),
        (df_s, "Group", "By Sector"),
        (df_c, "Group", "By Asset Class"),
    ]):
        ax.pie(df["Weight (%)"], labels=df[col], autopct="%1.1f%%",
               startangle=140, textprops={"color": "white", "fontsize": 8})
        ax.set_title(title, color="white")
        ax.set_facecolor(_PANEL)
    fig.patch.set_facecolor(_DARK)
    plt.tight_layout()
    path = CHARTS_DIR / "weights_breakdown.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=_DARK)
    plt.close()
    return str(path)


# ── CLI helpers ───────────────────────────────────────────────────────────────

def print_banner() -> None:
    console.print("""
[bold blue]╔════════════════════════════════════════════════════════╗[/bold blue]
[bold blue]║[/bold blue]  [bold white]a.s.r. Vermogensbeheer — Portfolio Assistant[/bold white]         [bold blue]║[/bold blue]
[bold blue]║[/bold blue]  [dim]Powered by GPT-4o · yfinance · MVC Architecture[/dim]     [bold blue]║[/bold blue]
[bold blue]╚════════════════════════════════════════════════════════╝[/bold blue]
""")

def print_info(msg: str) -> None:    console.print(f"[cyan]ℹ  {msg}[/cyan]")
def print_success(msg: str) -> None: console.print(f"[green]✅ {msg}[/green]")
def print_error(msg: str) -> None:   console.print(f"[red]❌ {msg}[/red]")
def print_thinking(msg: str) -> None: console.print(f"[yellow]🤔 {msg}[/yellow]")
def print_chart_saved(path: str) -> None: console.print(f"[dim]📊 Chart saved → {path}[/dim]")
def print_llm_response(text: str) -> None:
    console.print(Panel(text, title="[bold green]🤖 Assistant[/bold green]", border_style="green"))


# ── Stress Testing ────────────────────────────────────────────────────────────

def show_stress_test(results: list[dict]) -> None:
    """Display stress test results in Rich table."""
    if not results:
        console.print("[dim]No crisis periods found in data range.[/dim]")
        return
    for r in results:
        text = (f"[bold]{r['crisis']}[/bold] ({r['start']} → {r['end']})\n"
                f"  Portfolio drawdown: [red]{r['portfolio_drawdown_pct']:+.2f}%[/red]\n")
        for tk, dd in r.get("asset_drawdowns", {}).items():
            c = "green" if dd > -10 else "yellow" if dd > -25 else "red"
            text += f"  {tk}: [{c}]{dd:+.2f}%[/{c}]\n"
        console.print(Panel(text, title=f"📉 {r['crisis']}", border_style="red", expand=False))


def show_efficient_frontier(ef: dict) -> None:
    """Display efficient frontier summary."""
    text = (f"[bold]Efficient Frontier Analysis[/bold]\n\n"
            f"  Best Sharpe achievable: [cyan]{ef.get('best_sharpe', 0):.3f}[/cyan]\n"
            f"  Current portfolio: Return={ef.get('current_return', 0):.2f}%, Vol={ef.get('current_vol', 0):.2f}%\n\n"
            f"[bold]Optimal vs Current Weights:[/bold]\n")
    for tk in ef.get("optimal_weights", {}):
        opt = ef["optimal_weights"][tk]
        cur = ef.get("current_weights", {}).get(tk, 0)
        delta = opt - cur
        dc = "green" if delta > 0 else "red" if delta < 0 else "white"
        text += f"  {tk}: Optimal {opt:.1f}% | Current {cur:.1f}% | [{dc}]Δ {delta:+.1f}%[/{dc}]\n"
    console.print(Panel(text, title="📊 Efficient Frontier", border_style="cyan", expand=False))


def show_concentration_alerts(alerts: list[dict]) -> None:
    """Display concentration risk alerts."""
    if not alerts:
        console.print("[green]✓ No concentration risk alerts.[/green]")
        return
    for a in alerts:
        console.print(Panel(
            f"[bold yellow]⚠ {a['type'].title()} '{a['name']}'[/bold yellow]\n"
            f"  Weight: [bold]{a['weight_pct']:.1f}%[/bold] (limit: {a['limit_pct']:.0f}%)\n"
            f"  Excess: [red]{a['excess_pct']:.1f}%[/red]",
            border_style="yellow", expand=False))


def show_rolling_metrics(rm: dict) -> None:
    """Display rolling metrics summary."""
    if not rm:
        console.print("[dim]Not enough data for rolling metrics.[/dim]")
        return
    text = f"[bold]Rolling 12-Month Metrics[/bold]\n\n"
    if rm.get("sharpe"):
        text += f"  Latest Sharpe: {rm['sharpe'][-1]:.3f}\n"
        text += f"  Sharpe range: {min(rm['sharpe']):.3f} to {max(rm['sharpe']):.3f}\n"
    if rm.get("volatility"):
        text += f"  Latest Vol: {rm['volatility'][-1]:.2f}%\n"
    if rm.get("beta"):
        text += f"  Latest Beta: {rm['beta'][-1]:.3f}\n"
    console.print(Panel(text, title="📈 Rolling Metrics", border_style="blue", expand=False))


def show_rebalancing(suggestions: list[dict]) -> None:
    """Display rebalancing suggestions."""
    if not suggestions:
        console.print("[green]✓ Portfolio is well-balanced.[/green]")
        return
    text = "[bold]Rebalancing Suggestions[/bold]\n\n"
    for s in suggestions:
        c = "red" if s["action"] == "SELL" else "green"
        text += (f"  [{c}]{s['action']}[/{c}] {s['shares']} shares of "
                f"[bold]{s['ticker']}[/bold] (~€{s['value']:,.0f})\n"
                f"    {s['sector']}: {s['current_sector_pct']:.1f}% → {s['target_sector_pct']:.1f}%\n\n")
    console.print(Panel(text, title="⚖️ Rebalancing", border_style="cyan", expand=False))