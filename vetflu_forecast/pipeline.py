"""Сборка артефактов: ретроспектива (backtest) и живой прогноз -> JSON для сайта."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from . import config
from .backtest import run_backtest
from .config import HORIZON_M4, HORIZON_WAHIS, M4_N_SERIES, M4_ORIGINS, \
    M4_ORIGIN_STEP, SITE_DATA_DIR, WAHIS_ORIGINS, WAHIS_ORIGIN_STEP
from .data_m4 import load_m4_weekly
from .data_wahis import (fetch_events, load_snapshot, save_snapshot,
                         series_to_arrays, weekly_series)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "n/a"


def model_meta() -> dict:
    return {
        "repo": config.MODEL_REPO,
        "revision": config.MODEL_REVISION[:12],
        "max_context": config.MAX_CONTEXT,
        "license": "Apache-2.0 (weights 2.5)",
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_wahis_series() -> dict:
    """{key: pd.Series} — недельные ряды из снапшота (или свежего фетча, если его нет)."""
    try:
        events = load_snapshot()
    except FileNotFoundError:
        events = fetch_events()
        save_snapshot(events)
    return weekly_series(events)


def cmd_backtest(out_dir=None, m4_n: int | None = None) -> dict:
    out_dir = out_dir or SITE_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[backtest] WAHIS: свои ряды (мир + 4 флайвея)")
    wahis_pd = _get_wahis_series()
    wahis_items = [{"key": k, "y": s.to_numpy(dtype=float)} for k, s in wahis_pd.items()]
    panel_wahis = run_backtest(
        wahis_items, h=HORIZON_WAHIS, n_origins=WAHIS_ORIGINS,
        step=WAHIS_ORIGIN_STEP, with_hw=config.USE_HOLT_WINTERS["wahis"],
        label="wahis",
    )

    n_m4 = m4_n or M4_N_SERIES
    print(f"[backtest] M4 Weekly: выборка {n_m4} рядов")
    m4_items = [{"key": it["id"], "y": it["train"], "future": it["test"]}
                for it in load_m4_weekly(n_series=n_m4)]
    panel_m4 = run_backtest(
        m4_items, h=HORIZON_M4, n_origins=M4_ORIGINS,
        step=M4_ORIGIN_STEP, with_hw=config.USE_HOLT_WINTERS["m4"],
        label="m4",
    )

    doc = {
        "generated_at": _now(),
        "git_sha": git_sha(),
        "model": model_meta(),
        "method": {
            "scheme": "rolling-origin (M4 Weekly: 3 внутри train + 1 holdout на test)",
            "metric_primary": "MASE (сезонный, m=52)",
            "season_m": config.SEASON,
        },
        "panels": [
            {"label": "wahis", "title": "Свои ряды: ВПГП, WAHIS (недельные события)", **panel_wahis},
            {"label": "m4", "title": f"M4 Weekly, выборка {len(m4_items)} рядов", **panel_m4},
        ],
    }
    path = out_dir / "retrospective.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"[backtest] -> {path}")
    return doc


def cmd_forecast(out_dir=None) -> dict:
    from .model import forecast_batch

    out_dir = out_dir or SITE_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    series_pd = _get_wahis_series()
    keys = ["global"] + config.FLYWAYS
    h = HORIZON_WAHIS
    ctxs = [series_pd[k].to_numpy(dtype=float) for k in keys]
    point, quant = forecast_batch(ctxs, h)  # квантили: [...,0]=mean, [...,1]=q10, [...,9]=q90

    out_series = []
    for i, k in enumerate(keys):
        weeks, values = series_to_arrays(series_pd[k])
        tail = slice(max(0, len(weeks) - 156), len(weeks))  # ~3 года истории на график
        out_series.append({
            "key": k,
            "label": config.FLYWAY_LABELS[k],
            "history": {"weeks": weeks[tail], "values": [round(float(v), 2) for v in values[tail]]},
            "forecast": {
                "weeks": _next_weeks(weeks, h),
                "point": [round(float(v), 3) for v in point[i]],
                "q10": [round(float(v), 3) for v in quant[i, :, 1]],
                "q90": [round(float(v), 3) for v in quant[i, :, 9]],
            },
        })

    doc = {"generated_at": _now(), "git_sha": git_sha(), "model": model_meta(),
           "horizon_weeks": h, "data_source": "WOAH WAHIS (filtered-list, HPAI)",
           "series": out_series}
    path = out_dir / "forecast.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"[forecast] -> {path}")
    return doc


def _next_weeks(weeks: list[str], h: int) -> list[str]:
    import pandas as pd

    last = pd.Period(weeks[-1], freq="W")
    return [str(p) for p in pd.period_range(last + 1, last + h, freq="W")]
