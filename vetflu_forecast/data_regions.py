"""vet-heatmap: события по РФ -> недельные ряды (ФО × группа болезней) + сглаживание.

Источник — витрина vet-heatmap (data/outbreaks.json, CORS *): события уже
геокодированы (federal_district) и размечены группами болезней (disease_group).
Ряды — счёт событий в неделю (ISO-неделя, воскресенье, как в data_wahis),
сглаженные скользящим средним: на сырых счётчиках TimesFM в паритете с наивной
(панель wahis, Task 61), а на сглаженных ожидаем пользу — это проверяет
бэктест-панель regions, вердикт публикуем как есть.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import (DATA_DIR, HEATMAP_OUTBREAKS_URL, REGIONS_MIN_EVENTS,
                     REGIONS_SMOOTH_WEEKS, REGIONS_START)

SNAPSHOT_PATH = DATA_DIR / "outbreaks_heatmap.json"

DISTRICT_LABELS = {
    "ЦФО": "Центральный ФО",
    "СЗФО": "Северо-Западный ФО",
    "ЮФО": "Южный ФО",
    "СКФО": "Северо-Кавказский ФО",
    "ПФО": "Приволжский ФО",
    "УрФО": "Уральский ФО",
    "СФО": "Сибирский ФО",
    "ДФО": "Дальневосточный ФО",
}

GROUP_LABELS = {
    "Wildlife": "Дикие животные",
    "Ruminant": "Жвачные",
    "Swine": "Свиноводство",
    "Avian": "Птицеводство",
    "Multi-species": "Несколько видов",
    "Equine/Wildlife": "Лошади/дикие",
}


def group_label(g: str) -> str:
    return GROUP_LABELS.get(g, g)


def fetch_outbreaks(url: str | None = None, timeout: int = 90) -> dict:
    """Свежий outbreaks.json с витрины vet-heatmap (обновляется каждые 6 ч)."""
    r = requests.get(url or HEATMAP_OUTBREAKS_URL, timeout=timeout,
                     headers={"User-Agent": "vetflu-forecast-worker/1.0"})
    r.raise_for_status()
    return r.json()


def load_outbreaks_snapshot(path=None) -> dict:
    p = path or SNAPSHOT_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_outbreaks_snapshot(doc: dict, path=None) -> None:
    p = path or SNAPSHOT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))


def _smooth(s: pd.Series, window: int) -> pd.Series:
    """Трёхточечное скользящее среднее (trailing), хвост без NaN (min_periods=1)."""
    return s.rolling(window, min_periods=1).mean()


def region_series(doc: dict, start: str = REGIONS_START,
                  smooth_weeks: int = REGIONS_SMOOTH_WEEKS,
                  min_events: int = REGIONS_MIN_EVENTS,
                  ) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    """Недельные ряды по округам × группам болезней.

    Возвращает:
      series[key] = {"smooth": Series, "raw": Series}
        key = "ЦФО:Wildlife" (округ × группа) или "any:Wildlife" (вся Россия);
      meta[key] = {"district", "group", "label", "n_events"};
      districts = [{code, label, lat, lon, n_events}] — центроид округа по всем
        его событиям с координатами (независимо от min_events).

    Серии-комбо с < min_events событий не возвращаются (слишмо редкие для
    честного бэктеста); события без округа попадают только в национальные.
    """
    obs = doc.get("outbreaks") or []
    rows = []
    for o in obs:
        d = str(o.get("date") or "")
        if not d or d < start:
            continue
        try:
            ts = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append({
            "ts": ts,
            "fd": str(o.get("federal_district") or "").strip(),
            "group": str(o.get("disease_group") or "").strip(),
            "lat": o.get("lat"),
            "lon": o.get("lon"),
        })
    if not rows:
        raise ValueError("нет событий с валидной датой")

    df = pd.DataFrame(rows)
    df["week"] = df["ts"].dt.tz_localize(None).dt.to_period("W")
    idx = pd.period_range(pd.Period(start, freq="W"), df["week"].max(), freq="W")

    def counts(sub: pd.DataFrame) -> pd.Series:
        return sub.groupby("week").size().reindex(idx, fill_value=0.0).astype(float)

    series: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    # округ × группа
    for (fd, g), sub in df.groupby(["fd", "group"]):
        if not fd or not g or len(sub) < min_events:
            continue
        key = f"{fd}:{g}"
        raw = counts(sub)
        series[key] = {"smooth": _smooth(raw, smooth_weeks), "raw": raw}
        meta[key] = {
            "district": fd, "group": g,
            "label": f"{DISTRICT_LABELS.get(fd, fd)} — {group_label(g)}",
            "n_events": int(len(sub)),
        }

    # вся Россия × группа (включая события без округа)
    for g, sub in df.groupby("group"):
        if not g or len(sub) < min_events:
            continue
        key = f"any:{g}"
        raw = counts(sub)
        series[key] = {"smooth": _smooth(raw, smooth_weeks), "raw": raw}
        meta[key] = {
            "district": "any", "group": g,
            "label": f"Вся Россия — {group_label(g)}",
            "n_events": int(len(sub)),
        }

    if not series:
        raise ValueError("ни одна серия не прошла фильтр min_events")

    # центроиды округов (все события с координатами, независимо от min_events)
    districts: list[dict] = []
    for fd, sub in df[df["fd"] != ""].groupby("fd"):
        lat = pd.to_numeric(sub["lat"], errors="coerce").dropna()
        lon = pd.to_numeric(sub["lon"], errors="coerce").dropna()
        districts.append({
            "code": fd,
            "label": DISTRICT_LABELS.get(fd, fd),
            "lat": round(float(lat.mean()), 3) if len(lat) else None,
            "lon": round(float(lon.mean()), 3) if len(lon) else None,
            "n_events": int(len(sub)),
        })
    districts.sort(key=lambda d: -d["n_events"])

    return series, meta, districts

