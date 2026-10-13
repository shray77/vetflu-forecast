import numpy as np

from vetflu_forecast.baselines import drift, get_baselines, holt_winters, naive, snaive, sma


def test_naive_constant():
    out = naive(np.array([1.0, 2.0, 3.0]), 4)
    assert out.shape == (4,) and np.all(out == 3.0)


def test_snaive_repeats_last_season():
    ctx = np.arange(104, dtype=float)  # ровно 2 сезона по 52
    out = snaive(ctx, 13)
    assert np.allclose(out, ctx[-52:-52 + 13])
    long = snaive(ctx, 60)
    assert long.shape == (60,)
    assert np.allclose(long[:52], ctx[-52:])


def test_snaive_short_ctx_falls_back_to_naive():
    out = snaive(np.array([5.0, 7.0]), 3)
    assert np.all(out == 7.0)


def test_drift_slope():
    ctx = np.arange(100, dtype=float)
    out = drift(ctx, 3)
    assert np.allclose(out, [100.0, 101.0, 102.0])


def test_sma_window():
    ctx = np.array([1.0] * 6 + [5.0, 5.0])
    assert np.all(sma(ctx, 2, w=2) == 5.0)      # последние 2 = [5,5]
    assert np.all(sma(ctx, 2, w=8) == 2.0)      # (6*1 + 2*5)/8
    assert np.all(sma(ctx, 2, w=100) == (6.0 + 10.0) / 8.0)  # w режется по длине


def test_holt_winters_seasonal_series():
    t = np.arange(156)
    y = 10 + 0.1 * t + 8 * np.sin(2 * np.pi * t / 52) + np.random.default_rng(0).normal(0, 0.3, len(t))
    out = holt_winters(y, 13)
    assert out.shape == (13,) and np.all(np.isfinite(out))


def test_get_baselines_hw_toggle():
    assert "hw" not in get_baselines(False)
    assert "hw" in get_baselines(True)
