"""WOAH WAHIS: публичный POST API -> события ВПГП -> недельные ряды (мир + 4 флайвея).

Тот же источник, что скрейпит ВетКарта и использует vetflu-site (scripts/fetch_outbreaks.mjs).
Фильтр «Influenza A viruses of high pathogenicity» — клиентский (в API сортировка/пагинация).
Дедупликация по eventId (у события много рапортов; берём первую дату старта).
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import (CORRIDORS, DATA_DIR, FLYWAYS, WAHIS_API, WAHIS_MAX_PAGES,
                     WAHIS_PAGE_SIZE, WAHIS_SLEEP)

SNAPSHOT_PATH = DATA_DIR / "wahis_events.json"

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Referer": "https://wahis.woah.org/",
    "Origin": "https://wahis.woah.org",
}


def _is_hpai(e: dict) -> bool:
    d = str(e.get("disease", "")).lower()
    return "influenza" in d and "high pathogenicity" in d


def fetch_events(max_pages: int | None = None, verbose: bool = True) -> list[dict]:
    """Пагинация filtered-list по дате старта (desc) -> компактный список событий.

    Компактный формат: {id, country, disease, subtype, start, submission}.
    """
    max_pages = max_pages or WAHIS_MAX_PAGES
    out: list[dict] = []
    seen_ids = set()
    with requests.Session() as s:
        s.headers.update(HEADERS)
        for page in range(max_pages):
            body = {"pageNumber": page, "pageSize": WAHIS_PAGE_SIZE,
                    "sortOrder": "desc", "sortBy": "startDate"}
            r = s.post(WAHIS_API, json=body, timeout=60)
            r.raise_for_status()
            data = r.json()
            lst = data.get("list") or []
            if verbose:
                print(f"WAHIS page {page}: {len(lst)} rows (total so far {len(out)})")
            for row in lst:
                eid = row.get("eventId")
                if eid is None or eid in seen_ids:
                    continue  # дедуп: берём первую встреченную запись события (список отсортирован по дате)
                seen_ids.add(eid)
                if not _is_hpai(row):
                    continue
                out.append({
                    "id": int(eid),
                    "country": str(row.get("country") or "").strip(),
                    "disease": str(row.get("disease") or "").strip(),
                    "subtype": str(row.get("subType") or "").strip(),
                    "start": str(row.get("eventStartDate") or "")[:10] or None,
                    "submission": str(row.get("submissionDate") or "")[:10] or None,
                })
            if len(lst) < WAHIS_PAGE_SIZE:
                break
            time.sleep(WAHIS_SLEEP)
    if verbose:
        print(f"WAHIS: {len(seen_ids)} событий всего, HPAI = {len(out)}")
    return out


def load_snapshot(path=None) -> list[dict]:
    p = path or SNAPSHOT_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(events: list[dict], path=None) -> None:
    p = path or SNAPSHOT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, separators=(",", ":"))


def country_flyway_hits(country: str) -> list[tuple[str, float]]:
    """(флайвей, вес) для страны: primary 1.0, adjacent 0.4 — как в vetflu-site."""
    n = str(country).lower()
    hits = []
    for fid in FLYWAYS:
        c = CORRIDORS[fid]
        if any(s in n for s in c["primary"]):
            hits.append((fid, 1.0))
        elif any(s in n for s in c["adjacent"]):
            hits.append((fid, 0.4))
    return hits


def weekly_series(events: list[dict], week_start: str | None = None) -> dict[str, pd.Series]:
    """Недельные ряды (ISO-неделя, воскресенье): global = счёт событий, флайвеи = взвешенный счёт.

    Пустые недели = 0 (важно для честного бэктеста сезонности).
    """
    if not events:
        raise ValueError("пустой список событий WAHIS")
    rows = []
    for e in events:
        start = e.get("start")
        if not start:
            continue
        try:
            ts = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append((ts, e.get("country", "")))
    if not rows:
        raise ValueError("нет событий с валидной датой старта")
    df = pd.DataFrame(rows, columns=["ts", "country"])
    df["week"] = df["ts"].dt.tz_localize(None).dt.to_period("W")
    start_period = df["week"].min() if week_start is None else pd.Period(week_start, freq="W")
    idx = pd.period_range(start_period, df["week"].max(), freq="W")

    out = {"global": pd.Series(1.0, index=df["week"]).groupby(level=0).sum().reindex(idx, fill_value=0.0)}
    wcols = {fid: [] for fid in FLYWAYS}
    for country, week in zip(df["country"], df["week"]):
        for fid, w in country_flyway_hits(country):
            wcols[fid].append((week, w))
    for fid in FLYWAYS:
        if wcols[fid]:
            s = pd.DataFrame(wcols[fid], columns=["week", "w"]).groupby("week")["w"].sum()
        else:
            s = pd.Series(dtype=float)
        out[fid] = s.reindex(idx, fill_value=0.0)
    return out


def series_to_arrays(s: pd.Series) -> tuple[list[str], np.ndarray]:
    """(ISO-недели как строки YYYY-Wnn, значения float)."""
    return [str(p) for p in s.index], s.to_numpy(dtype=float)
