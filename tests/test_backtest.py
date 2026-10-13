import numpy as np
import pytest

from vetflu_forecast.backtest import (build_pairs, make_verdict, rolling_origins,
                                      run_backtest, season_strength)
from vetflu_forecast.config import MIN_CTX, SEASON


def _seasonal_series(key, weeks=260, seed=0):
    t = np.arange(weeks)
    rng = np.random.default_rng(seed)
    y = np.clip(20 + 15 * np.sin(2 * np.pi * t / SEASON) + rng.normal(0, 2, weeks), 0, None)
    return {"key": key, "y": y}


def test_rolling_origins():
    ends = rolling_origins(260, h=4, n_origins=6, step=8)
    assert ends == sorted(ends)
    assert all(e >= MIN_CTX for e in ends)
    assert ends[-1] == 260 - 4
    assert len(ends) == 6
    short = rolling_origins(110, h=4, n_origins=6, step=8)
    assert len(short) < 6  # короткий ряд -> меньше origins


def test_season_strength_strong_vs_noise():
    t = np.arange(208)
    strong = 20 + 15 * np.sin(2 * np.pi * t / SEASON)
    noise = np.random.default_rng(1).normal(0, 1, 208) + 50
    assert season_strength(strong) > 0.5
    assert season_strength(noise) < 0.2


def test_build_pairs_with_holdout():
    it = _seasonal_series("a")
    it["future"] = np.arange(13, dtype=float)
    pairs = build_pairs([it], h=13, n_origins=4, step=13)
    # 3 rolling внутри train + 1 holdout
    assert len(pairs) == 4
    assert pairs[-1]["actual"][0] == 0.0  # future использован как actual


def test_make_verdict_branches():
    v = make_verdict(0.9, "snaive", 1.0)
    assert v["status"] == "value"
    v = make_verdict(1.02, "snaive", 1.0)
    assert v["status"] == "parity"
    v = make_verdict(1.2, "snaive", 1.0)
    assert v["status"] == "no_value"


def test_run_backtest_with_mocked_model(monkeypatch):
    """Мок-модель (тупой snaive-клон) не должна ломать оркестр и агрегаты."""
    from vetflu_forecast.baselines import snaive

    def fake_forecast_batch(inputs, horizon):
        p = np.asarray([snaive(c, horizon) for c in inputs])
        q = np.zeros((len(inputs), horizon, 10))
        return p, q

    monkeypatch.setattr("vetflu_forecast.model.forecast_batch", fake_forecast_batch)
    items = [_seasonal_series(f"s{i}", seed=i) for i in range(3)]
    res = run_backtest(items, h=4, n_origins=2, step=8, use_timesfm=True, label="test")
    names = {r["model"] for r in res["summary"]}
    assert names == {"naive", "snaive", "drift", "sma8", "timesfm"}
    assert res["verdict"]["status"] in {"value", "parity", "no_value"}
    tfm = next(r for r in res["summary"] if r["model"] == "timesfm")
    sn = next(r for r in res["summary"] if r["model"] == "snaive")
    # мок повторяет snaive -> MASE совпадает
    assert abs(tfm["mase"] - sn["mase"]) < 1e-9
    assert res["n_pairs"] == 3 * 2


def test_run_backtest_buckets_present():
    items = [_seasonal_series(f"b{i}", seed=i) for i in range(3)]
    res = run_backtest(items, h=4, n_origins=2, step=8, use_timesfm=False)
    assert set(b["bucket"] for b in res["buckets"]) <= {"strong", "mid", "weak"}
