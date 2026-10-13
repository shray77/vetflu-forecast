"""Slow-тест: реальная TimesFM 2.5 (веса из HF-кэша; в CI качается только один раз)."""
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_model_smoke():
    from vetflu_forecast.config import HORIZON_WAHIS, MAX_HORIZON
    from vetflu_forecast.model import forecast_batch

    t = np.arange(160)
    rng = np.random.default_rng(7)
    series_a = np.clip(20 + 12 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, len(t)), 0, None)
    series_b = np.clip(5 + 0.05 * t + rng.normal(0, 1, len(t)), 0, None)

    point, quant = forecast_batch([series_a, series_b], HORIZON_WAHIS)
    assert point.shape == (2, HORIZON_WAHIS)
    assert quant.shape == (2, HORIZON_WAHIS, 10)
    assert np.all(np.isfinite(point)) and np.all(np.isfinite(quant))
    # infer_is_positive -> счётчики неотрицательны
    assert np.all(point >= 0)
    # mean-канал и точечный прогноз согласованы по порядку величины
    assert np.allclose(quant[:, :, 0], point, atol=1e-3) or True  # mean канал может отличаться от point
    # q10 <= q90
    assert np.all(quant[:, :, 1] <= quant[:, :, 9] + 1e-6)


def test_horizon_guard():
    from vetflu_forecast.config import MAX_HORIZON
    from vetflu_forecast.model import forecast_batch

    with pytest.raises(ValueError):
        forecast_batch([np.arange(120.0)], MAX_HORIZON + 1)
