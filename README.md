<div align="center">

# 🏦 a.s.r. Vermogensbeheer — Portfolio Assistant

**AI-powered investment portfolio analysis for institutional asset management**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/GPT--4o-Function_Calling-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-37_passed-22c55e?style=for-the-badge&logo=pytest&logoColor=white)](tests/)

*Built with MVC architecture · 11 agentic tools · Correlated Monte Carlo (Cholesky) · Real market data via yfinance*

---

</div>

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export OPENAI_API_KEY="sk-..."   # Linux/Mac
set OPENAI_API_KEY=sk-...        # Windows

# 3. Run
python main.py
```

> 💡 **Tip:** Run `python main.py --demo` to see all 11 tools in action automatically.

---

## 🏗️ Architecture (MVC)

```
asr-portfolio-assistant/
│
├── 📦 models/
│   └── portfolio.py           # Asset data, financial calculations, Monte Carlo
│
├── 🎨 views/
│   └── display.py             # Rich terminal tables, matplotlib charts
│
├── 🎮 controllers/
│   └── agent.py               # GPT-4o agent with 11 tools, agentic loop
│
├── 🧪 tests/
│   ├── test_portfolio.py      # 30 unit tests (mocked yfinance)
│   └── test_agent.py          # 7 integration tests
│
├── 📊 data/
│   └── portfolio.csv          # Sample portfolio (10 diversified assets)
│
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
└── .env.example               # Configuration template
```

| Layer | Responsibility |
|-------|---------------|
| **Model** | Asset storage, weight calculations, risk metrics (Sharpe, Sortino, Beta, VaR), Monte Carlo simulation, stress testing, efficient frontier |
| **View** | Rich terminal tables with color-coded P&L, matplotlib charts (correlation heatmap, returns bar, Monte Carlo fan chart) |
| **Controller** | GPT-4o agentic loop with `tool_choice="auto"`, multi-step chaining, 11 function-calling tools |

---

## 🚀 Usage

```bash
python main.py                                         # 💬 Interactive mode
python main.py -q "Show risk metrics for all assets"   # 🔍 Single query
python main.py --demo                                  # 🎬 Run all tasks
python main.py --model gpt-4o-mini                     # 💰 Use cheaper model
python main.py -p my_portfolio.csv                     # 📁 Custom portfolio
python main.py --log-level WARNING                     # 🔇 Hide debug logs
```

---

## ✅ The 4 Required Tasks

| # | Task | Example Query | Tool |
|---|------|--------------|------|
| 1 | 📈 Historical returns | `"Show historical yearly returns for AAPL"` | `show_historical_returns` |
| 2 | 📋 Portfolio overview | `"Show me the full portfolio overview"` | `show_portfolio_overview` |
| 3 | ⚖️ Weights & allocation | `"Show weights by asset, sector, and asset class"` | `show_weights` |
| 4 | 🎲 Monte Carlo simulation | `"Run the 15-year Monte Carlo simulation"` | `run_monte_carlo` |

---

## 🔧 Extended Tools (7 additional)

| Tool | Query | What it does |
|------|-------|-------------|
| 📉 Risk metrics | `"Show risk metrics for all assets"` | Sharpe, Sortino, Beta, Alpha, VaR, Calmar per asset + portfolio vs SPY |
| 🔗 Correlation | `"Show the correlation matrix"` | Pearson correlation + heatmap chart |
| 🔥 Stress test | `"Run a stress test"` | Portfolio drawdown during 2008, COVID, 2022 Rate Hikes |
| 📐 Efficient frontier | `"Show the efficient frontier"` | Markowitz optimisation with optimal weights |
| ⚠️ Concentration risk | `"Check for concentration risk"` | Flags sectors > 35% or assets > 25% |
| 📊 Rolling metrics | `"Show rolling Sharpe and volatility"` | 12-month rolling Sharpe, vol, beta |
| ⚖️ Rebalancing | `"How should I rebalance?"` | Concrete buy/sell suggestions with share counts |

> **Multi-step chaining**: Ask `"Check concentration risk and suggest how to fix it"` — the agent calls concentration → rebalancing automatically.

---

## 📐 Financial Metrics

### Per-Asset
| Metric | Description |
|--------|------------|
| CAGR | Compound annual growth rate (geometric) |
| Volatility | Annualised standard deviation of returns |
| Sharpe Ratio | Risk-adjusted return (excess return / volatility) |
| Sortino Ratio | Like Sharpe but uses downside deviation only |
| Beta | Sensitivity to SPY benchmark |
| Jensen's Alpha | Excess return over CAPM prediction |
| Treynor Ratio | Excess return per unit of systematic risk |
| Calmar Ratio | Return / max drawdown |
| VaR (95%, 99%) | Value at Risk — historical and parametric |
| Max Drawdown | Largest peak-to-trough decline |

### Portfolio-Level
All per-asset metrics **plus**: Tracking Error, Information Ratio, Diversification Ratio, Portfolio Beta — all benchmarked against **SPY**.

### Monte Carlo Simulation
- **Method**: Correlated multi-asset Geometric Brownian Motion via **Cholesky decomposition**
- **Paths**: 100,000 simulated paths over 15 years
- **Distributions**: Normal (default) or Student-t (fat tails)
- **Output**: Percentile fan chart, median, 5th/95th, CVaR, probability of profit

---

## 📁 Portfolio File Format

CSV with these columns:

```csv
ticker,name,sector,asset_class,quantity,purchase_price
AAPL,Apple Inc.,Technology,Equity,50,150.00
BND,Vanguard Total Bond Market ETF,Fixed Income,Bond,100,75.00
GLD,SPDR Gold Shares,Commodities,Commodity,20,170.00
```

JSON format is also supported.

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=models --cov=controllers
```

37 tests covering: CSV/JSON loading, weight calculations, risk metrics, Monte Carlo percentile ordering, stress test results, concentration alerts, efficient frontier, agent tool execution, error handling.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| 🤖 LLM | OpenAI GPT-4o with function calling |
| 📊 Market Data | yfinance (live prices, 5-year history) |
| 🧮 Computation | pandas, numpy, scipy |
| 📈 Charts | matplotlib, seaborn |
| 💻 CLI | Rich (tables, panels, progress) |
| 🧪 Testing | pytest with mocked yfinance |

---

<div align="center">

*Built for the a.s.r. Vermogensbeheer assignment*

</div>
