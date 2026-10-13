import numpy as np
import pytest

from vetflu_forecast.data_wahis import (country_flyway_hits, load_snapshot,
                                        save_snapshot, weekly_series)


def _ev(i, country, day):
    return {"id": 1000 + i, "country": country, "disease": "Influenza A viruses of high pathogenicity",
            "subtype": "2.3.4.4b", "start": f"2024-{day}", "submission": f"2024-{day}"}


def test_country_flyway_hits_weights():
    assert ("ea", 1.0) in country_flyway_hits("Russia")
    assert ("bsm", 1.0) in country_flyway_hits("Türkiye")
    w = dict(country_flyway_hits("France"))
    assert w.get("ea") == 0.4
    assert country_flyway_hits("Brazil") == []


def test_weekly_series_counts_and_zero_fill():
    events = [
        _ev(1, "Poland", "01-05"),
        _ev(1, "Poland", "01-06"),  # тот же eventId -> дедуп не в weekly_series (это fetch), тут оба рапорта события с одним id считаются дважды — тест проверяет, что ряд детерминирован
        _ev(2, "Germany", "01-07"),
        _ev(3, "Japan", "01-20"),
    ]
    s = weekly_series(events)
    assert set(s.keys()) == {"global", "ea", "bsm", "ca", "eaa"}
    g = s["global"]
    # первая неделя: 3 события, вторая: 0 (zero-fill!), третья: 1
    vals = g.to_numpy(dtype=float)
    assert vals[0] == 3.0 and vals[1] == 0.0 and vals[2] == 1.0
    assert np.all(np.diff(g.index.astype("int64")) == 1)  # недели подряд без дыр


def test_weekly_series_flyway_weights():
    events = [_ev(1, "Poland", "02-05"), _ev(2, "France", "02-05"), _ev(3, "Japan", "02-05")]
    s = weekly_series(events)
    assert s["ea"].iloc[0] == pytest.approx(1.0 + 0.4)
    assert s["eaa"].iloc[0] == pytest.approx(1.0)


def test_snapshot_roundtrip(tmp_path):
    events = [_ev(1, "Poland", "03-05")]
    p = tmp_path / "snap.json"
    save_snapshot(events, p)
    assert load_snapshot(p)[0]["id"] == 1001
