#!/usr/bin/env python3
"""
a.s.r. Vermogensbeheer — Portfolio Assistant CLI
Usage:
    python main.py                                    # interactive
    python main.py -q "Show risk metrics"             # single query
    python main.py --demo                             # run all tasks
    python main.py --model gpt-4o-mini -q "..."       # use different model
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.portfolio import Portfolio
from controllers.agent import PortfolioController
from views import display as view

DEMO_QUERIES = [
    "Show me the complete portfolio overview with all current values and P&L.",
    "Show the weights by asset, sector, and asset class.",
    "Show risk metrics including Sharpe, Sortino, Beta, and VaR for all assets and the portfolio.",
    "Show historical yearly returns for AAPL and MSFT.",
    "Show the correlation matrix between all assets.",
    "Run a stress test — how would my portfolio perform during historical crises?",
    "Check for any concentration risk in my portfolio.",
    "Run the 15-year Monte Carlo simulation.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="a.s.r. Portfolio Assistant")
    p.add_argument("--portfolio", "-p", default="data/portfolio.csv")
    p.add_argument("--query", "-q", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None, help="OpenAI model (gpt-4o, gpt-4o-mini)")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        view.print_error("No API key. Set OPENAI_API_KEY in .env or use --api-key.")
        sys.exit(1)

    view.print_banner()
    fp = Path(args.portfolio)
    if not fp.exists():
        view.print_error(f"Portfolio file not found: {fp}")
        sys.exit(1)

    view.print_info(f"Loading portfolio from {fp}…")
    try:
        portfolio = Portfolio.from_csv(fp) if fp.suffix == ".csv" else Portfolio.from_json(fp)
    except Exception as e:
        view.print_error(f"Failed to load: {e}")
        sys.exit(1)
    view.print_success(f"Loaded {len(portfolio.assets)} assets · Total value: €{portfolio.total_value:,.2f}")

    controller = PortfolioController(portfolio, api_key=api_key, model=args.model)

    if args.demo:
        view.console.print("\n[bold yellow]🎬 Running demo…[/bold yellow]\n")
        for q in DEMO_QUERIES:
            view.console.rule(f"[bold]{q[:80]}[/bold]")
            controller.chat(q)
            view.console.print()
        return

    if args.query:
        controller.chat(args.query)
        return

    # Interactive
    view.console.print("\n[dim]Commands: exit, clear[/dim]\n")
    while True:
        try:
            user_input = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            view.console.print("\n[dim]Goodbye![/dim]")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if user_input.lower() == "clear":
            controller.conversation_history.clear()
            view.print_success("History cleared.")
            continue
        controller.chat(user_input)
        view.console.print()


if __name__ == "__main__":
    main()
