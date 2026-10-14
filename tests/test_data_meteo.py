"""Тесты data_meteo: THI-формула и недельная агрегация (без сети)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vetflu_forecast.data_meteo import thi_nrc, weekly_series_from_daily


def test_thi_formula_matches_vet_meteo():
    # THI(25, 50) = 71.775 (сверка с формулой vet-meteo/lib/models/thi.ts)
    assert abs(thi_nrc(25, 50) - 71.775) < 1e-9
    # RH=0 -> поправочный множитель 0.55: THI(26,0) = 78.8 - 0.55*20.8 = 67.36
    assert abs(thi_nrc(26, 0) - 67.36) < 1e-9


def test_weekly_aggregation_shape_and_values():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")
    df = pd.DataFrame({"date": idx, "t": np.linspace(0, 13, 14), "rh": 50.0})
    tw, thw = weekly_series_from_daily(df)
    # 14 дней -> 2 недели W-SUN (1-7, 8-14)
    assert len(tw) == 2 and len(thw) == 2
    assert np.isfinite(tw.to_numpy()).all()
    # THI монотонен по t при фиксированном RH
    assert (thw.to_numpy()[1] > thw.to_numpy()[0]) == (tw.to_numpy()[1] > tw.to_numpy()[0])


def test_thi_consistent_with_manual():
    t, rh = -5.0, 80.0
    manual = (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26)
    assert abs(thi_nrc(t, rh) - manual) < 1e-9
