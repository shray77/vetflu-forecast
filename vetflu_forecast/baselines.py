"""Простые статистические бейзлайны — то, с чем сравниваем TimesFM.

Идея ретроспективы: foundation-модель имеет смысл, только если бьёт дешёвую статистику
на наших рядах. Бейзлайны честные: сезонная наивная (повтор последнего сезона) —
классический сильный противник на недельных данных.
"""
from __future__ import annotations

import numpy as np

from .config import SEASON


def naive(ctx, h: int) -> np.ndarray:
    """Последнее значение, повторенное h раз."""
    return np.repeat(float(np.asarray(ctx, dtype=float)[-1]), h)


def snaive(ctx, h: int, m: int = SEASON) -> np.ndarray:
    """Повтор последнего полного сезона (52 недели) необходимое число раз."""
    ctx = np.asarray(ctx, dtype=float)
    if len(ctx) < m:
        return naive(ctx, h)
    tail = ctx[-m:]
    reps = int(np.ceil(h / m))
    return np.tile(tail, reps)[:h]


def drift(ctx, h: int) -> np.ndarray:
    """Наивный дрейф: линия через первое и последнее значения."""
    ctx = np.asarray(ctx, dtype=float)
    if len(ctx) < 2:
        return naive(ctx, h)
    slope = (ctx[-1] - ctx[0]) / (len(ctx) - 1)
    return ctx[-1] + slope * np.arange(1, h + 1)


def sma(ctx, h: int, w: int = 8) -> np.ndarray:
    """Среднее последних w точек."""
    ctx = np.asarray(ctx, dtype=float)
    w = min(w, len(ctx))
    return np.repeat(float(ctx[-w:].mean()), h)


def holt_winters(ctx, h: int, m: int = SEASON) -> np.ndarray:
    """Аддитивный Хольт-Уинтерс (statsmodels); при сбое деградируем в snaive."""
    ctx = np.asarray(ctx, dtype=float)
    if len(ctx) < 2 * m:
        return snaive(ctx, h, m)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        fit = ExponentialSmoothing(
            ctx, trend="add", seasonal="add", seasonal_periods=m,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.asarray(fit.forecast(h), dtype=float)
    except Exception:
        return snaive(ctx, h, m)


def get_baselines(with_hw: bool = False) -> dict:
    """Словарь «имя -> callable(ctx, h)». HW включается только на своих рядах."""
    models = {"naive": naive, "snaive": snaive, "drift": drift, "sma8": sma}
    if with_hw:
        models["hw"] = holt_winters
    return models
