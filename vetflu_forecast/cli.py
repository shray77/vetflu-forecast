"""CLI: python -m vetflu_forecast.cli {fetch|backtest|forecast|all}"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vetflu-forecast",
                                description="TimesFM 2.5 пилот: worker + ретроспектива")
    p.add_argument("command", choices=["fetch", "backtest", "forecast", "all"])
    p.add_argument("--out", default=None, help="каталог для JSON (по умолчанию site/data)")
    p.add_argument("--m4-n", type=int, default=None,
                   help="размер выборки M4 Weekly для бэктеста (по умолчанию 120)")
    args = p.parse_args(argv)
    from pathlib import Path

    out = Path(args.out) if args.out else None

    if args.command == "fetch":
        from .data_wahis import fetch_events, save_snapshot

        events = fetch_events()
        save_snapshot(events)
        print(f"снапшот: {len(events)} HPAI-событий")
        return 0

    if args.command in ("backtest", "all"):
        from .pipeline import cmd_backtest

        cmd_backtest(out_dir=out, m4_n=args.m4_n)
    if args.command in ("forecast", "all"):
        from .pipeline import cmd_forecast

        cmd_forecast(out_dir=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
