"""Command-line interface.

Examples
--------
    stablecoin-risk run all                 # full pipeline -> reports/
    stablecoin-risk run clean liquidity     # selected steps
    stablecoin-risk collect-daily --start 2020-01-01 --end 2026-02-20
"""

from __future__ import annotations

import argparse
import sys

from stablecoin_risk import config


def main(argv: list[str] | None = None) -> int:
    from stablecoin_risk.pipeline import STEPS, run

    parser = argparse.ArgumentParser(prog="stablecoin-risk", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run analysis steps")
    p_run.add_argument("steps", nargs="+", choices=["all", *STEPS], help="steps to run, or 'all'")

    p_col = sub.add_parser("collect-daily", help="download the daily BTC/ETH/USDC/USDT panel from public APIs")
    p_col.add_argument("--start", default="2020-01-01")
    p_col.add_argument("--end", required=True)
    p_col.add_argument("--out", default=str(config.RAW_DIR / "daily" / "crypto_ohlcv_daily_collected.csv"))

    args = parser.parse_args(argv)
    if args.command == "run":
        results = run(None if "all" in args.steps else args.steps)
        skipped = [k for k, v in results.items() if isinstance(v, str)]
        print(f"\nWrote {config.REPORTS_DIR / 'RESULTS.md'}" + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    elif args.command == "collect-daily":
        from stablecoin_risk.data.collectors import collect_daily_panel

        panel = collect_daily_panel(["BTC/USD", "ETH/USD", "USDC/USD", "USDT/USD"], args.start, args.end)
        panel.to_csv(args.out)
        print(f"Wrote {len(panel):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
