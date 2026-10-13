"""Тесты data_regions: раскладка по округам × группам, недели, сглаживание, фильтры."""
from __future__ import annotations

import pandas as pd
import pytest

from vetflu_forecast.data_regions import region_series


def _doc(outbreaks: list[dict]) -> dict:
    return {"updated": "2026-10-13T00:00:00Z", "sources": ["test"], "total_outbreaks": len(outbreaks),
            "outbreaks": outbreaks}


def _sundays(n: int, start: str = "2024-01-07") -> list[str]:
    return [str(d.date()) for d in pd.date_range(start, periods=n, freq="7D")]


def _ev(date: str, fd: str, group: str, lat=50.0, lon=40.0) -> dict:
    return {"id": 1, "disease_key": "test", "disease": "Тест", "disease_group": group,
            "region": "Тестовая область", "date": date, "species": "Other", "cases": 1,
            "deaths": 0, "status": "Resolved", "source": "wahis", "source_url": "",
            "notes": "", "region_inferred": False, "federal_district": fd,
            "lat": lat, "lon": lon, "geocode_source": "test"}


@pytest.fixture()
def doc():
    dates = _sundays(30)  # 30 воскресений 2024 года
    outbreaks = (
        [_ev(d, "ЦФО", "Wildlife", lat=50.0 + i * 0.1, lon=40.0) for i, d in enumerate(dates)]
        + [_ev(d, "ЮФО", "Avian", lat=45.0, lon=42.0) for d in dates[:3]]
        + [_ev(d, "", "Wildlife", lat=55.0, lon=50.0) for d in dates[:2]]
        + [_ev("2022-12-04", "ЦФО", "Wildlife", lat=50.0, lon=40.0)]  # до REGIONS_START
    )
    return _doc(outbreaks)


def test_keys_and_min_events(doc):
    series, meta, districts = region_series(doc)
    assert "ЦФО:Wildlife" in series          # 30 событий
    assert "any:Wildlife" in series          # национальная, включая события без округа
    assert "any:Avian" not in series         # 3 события < min_events — фильтр действует и на национальные
    assert "ЮФО:Avian" not in series         # 3 события < REGIONS_MIN_EVENTS (24)
    assert series["ЦФО:Wildlife"]["raw"].sum() == 30.0
    assert series["any:Wildlife"]["raw"].sum() == 32.0  # 30 ЦФО + 2 без округа
    assert meta["any:Wildlife"]["district"] == "any"
    assert meta["ЦФО:Wildlife"]["label"] == "Центральный ФО — Дикие животные"
    # события без округа не порождают серию с пустым округом
    assert not any(k.startswith(":") for k in series)


def test_min_events_override(doc):
    series, _, _ = region_series(doc, min_events=3)
    assert "ЮФО:Avian" in series
    series2, _, _ = region_series(doc, min_events=31)
    assert "ЦФО:Wildlife" not in series2
    assert "any:Wildlife" in series2  # 32 события


def test_weekly_index_and_zeros(doc):
    series, _, _ = region_series(doc)
    raw = series["ЦФО:Wildlife"]["raw"]
    # индекс начинается с REGIONS_START (2023-01-01) и до последней недели данных
    assert str(raw.index[0]) == "2022-12-26/2023-01-01" or str(raw.index[0]).startswith("2022-12-26")
    assert raw.index[-1] == pd.Period("2024-01-07", freq="W") + 29
    # события только в 30 недель — остальные нули
    assert (raw > 0).sum() == 30


def test_smoothing_trailing_mean(doc):
    series, _, _ = region_series(doc)
    raw = series["ЦФО:Wildlife"]["raw"].to_numpy()
    sm = series["ЦФО:Wildlife"]["smooth"].to_numpy()
    assert sm[0] == pytest.approx(raw[0])          # min_periods=1
    assert sm[3] == pytest.approx(raw[:4].mean())  # полное окно = 4
    tail = raw[-4:].mean()
    assert sm[-1] == pytest.approx(tail)


def test_district_centroids(doc):
    _, _, districts = region_series(doc)
    by_code = {d["code"]: d for d in districts}
    # центроид ЦФО = среднее по 30 событиям (lat 50.0..52.9, lon 40)
    assert by_code["ЦФО"]["lat"] == pytest.approx(50.0 + 0.1 * 29 / 2, abs=0.01)
    assert by_code["ЦФО"]["lon"] == pytest.approx(40.0, abs=0.01)
    assert by_code["ЦФО"]["label"] == "Центральный ФО"
    assert by_code["ЦФО"]["n_events"] == 30  # событие 2022 года не считается
    assert "any" not in by_code


def test_events_before_start_excluded(doc):
    series, meta, districts = region_series(doc)
    # 2022-12-04 отсечена: ЦФО = 30 событий, не 31
    assert meta["ЦФО:Wildlife"]["n_events"] == 30
    assert series["ЦФО:Wildlife"]["raw"].sum() == 30.0


def test_empty_doc_raises():
    with pytest.raises(ValueError):
        region_series(_doc([]))
