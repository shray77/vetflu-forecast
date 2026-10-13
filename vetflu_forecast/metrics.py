"""Метрики точности прогноза: MASE (сезонный), WAPE, sMAPE."""
from __future__ import annotations

import numpy as np


def mase(actual, pred, context, m: int = 52) -> float:
    """MASE со сезонным масштабом: средняя |ошибка| / средняя |сезонная разница| на контексте."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    c = np.asarray(context, dtype=float)
    if len(c) <= m:
        scale = np.mean(np.abs(np.diff(c))) if len(c) > 1 else 1.0
    else:
        scale = np.mean(np.abs(c[m:] - c[:-m]))
    if not np.isfinite(scale) or scale < 1e-12:
        alt = np.mean(np.abs(np.diff(c))) if len(c) > 1 else 1.0
        scale = alt if alt > 1e-12 else 1.0
    return float(np.mean(np.abs(a - p)) / scale)


def wape(actual, pred) -> float:
    """WAPE = sum|e| / sum|y| (в [0, inf); на нулевых рядах возвращаем 0)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = np.abs(a).sum()
    if denom < 1e-12:
        return 0.0
    return float(np.abs(a - p).sum() / denom)


def smape(actual, pred) -> float:
    """sMAPE в процентах; пары с нулевым знаменателем исключаются."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    denom = (np.abs(a) + np.abs(p)) / 2.0
    mask = denom > 1e-12
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(a - p)[mask] / denom[mask]) * 100.0)
