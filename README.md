# a.s.r. Vermogensbeheer — Portfolio Assistant

LLM-powered investment portfolio analysis tool. CLI interface, MVC architecture, GPT-4o function calling, real market data via yfinance.

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # Linux/Mac
set OPENAI_API_KEY=sk-...        # Windows
python main.py --demo
```

## Architecture (MVC)

```
asr-portfolio-assistant/
├── models/
│   └── portfolio.py       # Data + all financial calculations
├── views/
│   └── display.py         # Rich terminal tables + matplotlib charts
├── controllers/
│   └── agent.py           # GPT-4o agent (10 tools, agentic loop)
├── tests/
│   ├── test_portfolio.py  # 30 unit tests
│   └── test_agent.py      # 7 integration tests
├── data/
│   └── portfolio.csv      # Sample portfolio (10 assets)
├── charts/                # Auto-generated chart PNGs
├── main.py                # CLI entry point
├── requirements.txt
└── .env.example
```

## Usage

```bash
python main.py                                         # interactive mode
python main.py -q "Show risk metrics for all assets"   # single query
python main.py --demo                                  # run all tasks
python main.py --model gpt-4o-mini                     # use cheaper model
python main.py -p my_portfolio.csv                     # custom portfolio
```

## The 4 Required Tasks

| # | Task | Example Query |
|---|------|--------------|
| 1 | Historical returns | "Show historical monthly, quarterly and yearly returns for AAPL" |
| 2 | Portfolio overview | "Show me the full portfolio overview" |
| 3 | Weights | "Show weights by asset, sector, and asset class" |
| 4 | Monte Carlo | "Run the 15-year Monte Carlo simulation" |

## Extended Tools (6 additional)

| Tool | Query |
|------|-------|
| Risk metrics | "Show risk metrics and Sharpe ratios for all assets" |
| Correlation | "Show the correlation matrix between all assets" |
| Stress test | "Run a stress test for historical crises" |
| Efficient frontier | "Show the efficient frontier and optimal weights" |
| Concentration risk | "Check for concentration risk" |
| Rolling metrics | "Show rolling 12-month Sharpe and volatility" |

## Financial Metrics

**Per-asset**: CAGR, Volatility, Sharpe, Sortino, Beta, Alpha, Treynor, Calmar, VaR (95%/99%), Max Drawdown

**Portfolio**: All above + Tracking Error, Information Ratio, Diversification Ratio vs SPY

**Simulation**: Correlated multi-asset GBM (Cholesky), 100K paths, Normal + Student-t

## Portfolio File Format

CSV with columns: `ticker`, `name`, `sector`, `asset_class`, `quantity`, `purchase_price`

## Testing

```bash
pytest tests/ -v --cov=models --cov=controllers
```

## Tech Stack

Python · OpenAI GPT-4o · yfinance · pandas · numpy · scipy · matplotlib · Rich
