"""Обёртка TimesFM 2.5 (200M, Apache-2.0): ленивая загрузка + батчевый forecast.

API пакета timesfm==2.0.2:
  model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=..., torch_compile=False)
  model.compile(timesfm.ForecastConfig(...))
  point, quant = model.forecast(horizon=h, inputs=[...])
  point (B, H); quant (B, H, 10) = [mean, q10..q90].

Модель качается в стандартный кэш HF (~/.cache/huggingface) — именно эту папку
кэширует actions/cache в CI, чтобы воркер не качал 925 МБ каждый запуск.
"""
from __future__ import annotations

import numpy as np

from .config import (MAX_CONTEXT, MAX_HORIZON, MODEL_REPO, MODEL_REVISION,
                     PER_CORE_BATCH_SIZE)

_model = None


def get_model():
    """Синглтон модели (лениво; torch/timesfm импортируются только здесь)."""
    global _model
    if _model is not None:
        return _model
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, torch_compile=False,
    )
    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT,
        max_horizon=MAX_HORIZON,
        normalize_inputs=True,
        per_core_batch_size=PER_CORE_BATCH_SIZE,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,     # наши ряды — счёт событий (>=0)
        fix_quantile_crossing=True,
    ))
    _model = model
    return _model


def forecast_batch(inputs: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Батчевый прогноз. Возвращает (point, quantiles), обрезанные до len(inputs).

    quantiles: (B, H, 10) — [mean, q10, q20, ..., q90].
    """
    if horizon > MAX_HORIZON:
        raise ValueError(f"horizon {horizon} > MAX_HORIZON {MAX_HORIZON}")
    model = get_model()
    inputs = [np.asarray(x, dtype=float) for x in inputs]
    point, quant = model.forecast(horizon=horizon, inputs=inputs)
    return point[: len(inputs)], quant[: len(inputs)]
