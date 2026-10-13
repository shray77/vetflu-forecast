import numpy as np

from vetflu_forecast.metrics import mase, smape, wape


def test_mase_zero_error():
    ctx = np.arange(200, dtype=float)
    y = ctx[-5:]
    assert mase(y, y, ctx, m=52) == 0.0


def test_mase_known_value():
    ctx = np.arange(150, dtype=float)  # сезонная разница = 52
    actual = np.array([160.0, 180.0])
    pred = np.array([150.0, 150.0])
    expected = np.mean(np.abs(actual - pred)) / 52.0
    assert abs(mase(actual, pred, ctx, m=52) - expected) < 1e-9


def test_mase_degenerate_scale_fallback():
    ctx = np.ones(150, dtype=float)  # все разницы нулевые -> фолбэк
    actual = np.array([2.0, 2.0])
    pred = np.array([1.0, 1.0])
    assert mase(actual, pred, ctx, m=52) > 0


def test_wape_basic():
    actual = np.array([10.0, 0.0])
    pred = np.array([0.0, 0.0])
    assert abs(wape(actual, pred) - 1.0) < 1e-12
    assert wape(np.zeros(5), np.ones(5)) == 0.0  # вырожденный ряд -> 0


def test_smape_bounds():
    actual = np.array([100.0, 1.0, 0.0])
    pred = np.array([0.0, 1.0, 0.0])
    v = smape(actual, pred)
    assert 0.0 <= v <= 100.0
    assert smape(actual, actual) == 0.0
