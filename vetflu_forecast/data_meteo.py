"""Недельные метео-ряды по флайвеям: эксперимент «погода через foundation-модель».

Ради интереса (решение экосистемы: погоду в проде НЕ предсказываем, THI — справка).
Вопрос эксперимента: что вообще покажет foundation-модель на погодных недельных
рядах против persistence (naive) — классического чемпиона краткосрочной метеоры?

Данные: Open-Meteo archive API (без ключа, CC-BY) — дневные средние t/RH по
центроидам флайвеев (среднее координат path из data/migration.json vetflu-site).
Агрегация: недельные средние W-SUN (конвенция экосистемы) + THI (NRC/Thom) по
недельным средним — ВНИМАНИЕ: THI из средних занижает пиковый стресс, для
прод-расчётов в vet-meteo считается по часовым данным; тут только для эксперимента.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_DIR

FLYWAYS_JSON = "https://raw.githubusercontent.com/shray77/vetflu-site/main/data/migration.json"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
METEO_START = "2015-01-01"   # ~10 лет контекста (как MAX_CONTEXT TimesFM, 512 нед)
METEO_CACHE = DATA_DIR / "meteo_cache.json"

FLYWAY_LABELS = {"ea": "Восточно-Атлантический", "bsm": "Черноморско-Средиземноморский",
                 "ca": "Центральноазиатский", "eaa": "Восточноазиатско-Австралазийский"}


def thi_nrc(t: float, rh: float) -> float:
    """THI (NRC/Thom): (1.8t+32) - (0.55 - 0.0055*RH) * (1.8t - 26). Тот же, что в vet-meteo."""
    f = 1.8 * t + 32
    return f - (0.55 - 0.0055 * rh) * (1.8 * t - 26)


def fetch_flyway_centroids() -> dict[str, tuple[float, float]]:
    """{fw_id: (lat, lon)} — среднее координат пути из migration.json vetflu-site.

    path — плоский список точек; у eaa путь многосегментный (paths = список списков).
    """
    r = requests.get(FLYWAYS_JSON, timeout=30)
    r.raise_for_status()
    doc = r.json()
    out = {}
    for fw in doc["flyways"]:
        pts = list(fw.get("path") or [])
        for seg in fw.get("paths") or []:
            pts.extend(seg)
        if not pts:
            continue
        lats = [float(p[0]) for p in pts]
        lons = [float(p[1]) for p in pts]
        out[fw["id"]] = (sum(lats) / len(lats), sum(lons) / len(lons))
    return out


NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def _fetch_daily_power(lat: float, lon: float) -> pd.DataFrame:
    """NASA POWER (MERRA-2 реанализ, без ключа): T2M/RH2M.

    Основной провайдер: у Open-Meteo дневная квота на IP (сандбокс/раннер её
    легко исчерпывают), POWER щедрее. Минусы: отставание ~1-2 нед от сейчас
    (для бэктеста неважно), реанализ вместо станций (для эксперимента ок).
    """
    start = METEO_START.replace("-", "")
    end = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    params = {
        "parameters": "T2M,RH2M", "community": "AG",
        "latitude": lat, "longitude": lon,
        "start": start, "end": end, "format": "JSON",
    }
    r = requests.get(NASA_POWER_URL, params=params, timeout=120)
    r.raise_for_status()
    props = r.json()["properties"]["parameter"]
    t = props["T2M"]
    rh = props["RH2M"]
    df = pd.DataFrame({
        "date": pd.to_datetime([k for k in t.keys() if t[k] is not None], format="%Y%m%d"),
        "t": [v for v in t.values() if v is not None],
        "rh": [rh[k] for k in t.keys() if t[k] is not None],
    })
    # -999 = нет данных в POWER
    df = df[(df["t"] > -100) & (df["rh"] > -100)]
    return df.dropna()


def fetch_daily(lat: float, lon: float, retries: int = 3) -> pd.DataFrame:
    """Дневные средние t/RH с METEO_START по точке.

    Порядок: NASA POWER (щедрые лимиты) -> Open-Meteo archive (может упереться
    в дневную квоту IP: «Daily API request limit exceeded»).

    Open-Meteo душит 429 (per-minute лимит) — ретраи с экспоненциальной паузой.
    """
    import time

    try:
        return _fetch_daily_power(lat, lon)
    except Exception as e:  # noqa: BLE001 — фолбэк на любой сбой POWER
        print(f"[meteo] NASA POWER не ответил ({e}), фолбэк на Open-Meteo")

    params = {
        "latitude": lat, "longitude": lon, "start_date": METEO_START,
        "end_date": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        "daily": "temperature_2m_mean,relative_humidity_2m_mean", "timezone": "UTC",
    }
    last: Exception | None = None
    for k in range(retries):
        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=60)
            if r.status_code == 429:
                pause = 30 * (k + 1)
                print(f"[meteo] 429, пауза {pause} c ({k + 1}/{retries})")
                time.sleep(pause)
                continue
            r.raise_for_status()
            d = r.json()["daily"]
            return pd.DataFrame({
                "date": pd.to_datetime(d["time"]),
                "t": d["temperature_2m_mean"],
                "rh": d["relative_humidity_2m_mean"],
            }).dropna()
        except requests.HTTPError as e:
            last = e
            if e.response is not None and e.response.status_code == 429:
                time.sleep(30 * (k + 1))
                continue
            raise
    raise RuntimeError(f"Ни POWER, ни Open-Meteo не ответили: {last}")


def weekly_series_from_daily(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Недельные средние (W-SUN): температура и THI по недельным средним."""
    df = df.set_index("date")
    tw = df["t"].resample("W-SUN").mean().dropna()
    thw = df.apply(lambda r: thi_nrc(r["t"], r["rh"]), axis=1).resample("W-SUN").mean().dropna()
    return tw, thw


def build_meteo_series(force: bool = False) -> dict[str, pd.Series]:
    """{key: pd.Series}; key = '<fw>.temp' | '<fw>.thi'. Кэш в data/meteo_cache.json."""
    if not force and METEO_CACHE.exists():
        raw = json.loads(METEO_CACHE.read_text())
        return {k: pd.Series(v) for k, v in raw.items()}

    cents = fetch_flyway_centroids()
    out: dict[str, pd.Series] = {}
    for fw, (lat, lon) in sorted(cents.items()):
        df = fetch_daily(lat, lon)
        tw, thw = weekly_series_from_daily(df)
        out[f"{fw}.temp"] = tw
        out[f"{fw}.thi"] = thw
        print(f"[meteo] {fw} ({lat:.1f}, {lon:.1f}): {len(tw)} недель")
    METEO_CACHE.write_text(json.dumps(
        {k: [None if pd.isna(v) else round(float(v), 4) for v in s]
         for k, s in out.items()}))
    return out


def load_meteo_items(force: bool = False) -> list[dict]:
    """items для бэктеста: {'key', 'y'} — обрезка до 512 нед (MAX_CONTEXT)."""
    from .config import MAX_CONTEXT

    series = build_meteo_series(force=force)
    items = []
    for k, s in sorted(series.items()):
        y = s.to_numpy(dtype=float)[-MAX_CONTEXT:]
        items.append({"key": k, "y": y})
    return items
