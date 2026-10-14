"""Экспериментальные foundation-модели для A/B против TimesFM 2.5 (см. scripts/ab_run.py).

Контракт тот же, что у model.forecast_batch: forecast_batch(inputs, horizon) ->
(point (B, H), quantiles|None). Зависимости опциональны и ставятся отдельно:

  pip install chronos-forecasting          # Chronos-2, Apache-2.0 (amazon/chronos-2)
  pip install --no-deps toto-2 toto-models # Toto 2.0, Apache-2.0 (Datadog/Toto-2.0-*)
  pip install gluonts jaxtyping dd-unit-scaling lightning

Важно (2026-10): transformers нужен 4.x (5.x требует torch>=2.7 из-за
torch.accelerator, а репо пинит torch 2.5.1 CPU). Обе модели — Apache-2.0,
в отличие от TimesFM 3.0 (non-commercial, не используем — см. README).

Честность сравнения: модели получают ТЕ же пары (ctx, actual) из backtest.build_pairs,
тот же протокол rolling-origin и те же метрики metrics.py. Point-прогноз у Chronos-2
и Toto — медиана (q50), у TimesFM — point-голова; это штатный режим каждой модели.
"""
from __future__ import annotations

import numpy as np

CHRONOS2_REPO = "amazon/chronos-2"
TOTO2_REPO = "Datadog/Toto-2.0-4m"   # 4M параметров (~16 МБ) — семейство Toto 2.0 (u-muP)
TOTO_CTX = 512                        # как MAX_CONTEXT TimesFM — одинаковый контекст у всех
TOTO_DECODE_BLOCK = 768               # из README Toto 2.0 (quick start)

_chronos2 = None
_toto2 = None


def _set_threads(n: int = 2) -> None:
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def get_chronos2():
    """Синглтон Chronos2Pipeline (лениво; chronos/transformers импортируются тут)."""
    global _chronos2
    if _chronos2 is not None:
        return _chronos2
    import torch
    from chronos import Chronos2Pipeline

    _set_threads()
    _chronos2 = Chronos2Pipeline.from_pretrained(
        CHRONOS2_REPO, device_map="cpu", torch_dtype=torch.float32,
    )
    return _chronos2


def chronos2_batch(inputs: list[np.ndarray], horizon: int, batch_size: int = 8) -> np.ndarray:
    """Point-прогноз Chronos-2 (медиана q50): (B, H).

    predict() -> list[(n_variates, n_quantiles, H)]. ВАЖНО: уровней 21
    (0.01..0.99, шаг 0.05 в середине) — индекс медианы ищем по массиву уровней,
    НЕ хардкодом (история: при 9 квантилях брали idx 4 = q0.20 и занижали прогноз).
    """
    pipe = get_chronos2()
    ctxs = [np.asarray(x, dtype=float) for x in inputs]
    out = pipe.predict(inputs=ctxs, prediction_length=horizon, batch_size=batch_size)
    med_idx = _chronos2_median_idx(pipe)
    pts = np.stack([t[0, med_idx, :].detach().cpu().numpy() for t in out])
    if pts.shape != (len(inputs), horizon):
        raise ValueError(f"chronos2: ждали {(len(inputs), horizon)}, получили {pts.shape}")
    return pts


def _chronos2_median_idx(pipe) -> int:
    """Индекс квантиля 0.5 в levels модели (у Chronos-2 их 21)."""
    inner = getattr(getattr(pipe, "model", None), "chronos2", None) or getattr(pipe, "model", None)
    levels = getattr(inner, "quantiles", None)
    if levels is not None:
        import numpy as _np

        return int(_np.argmin(_np.abs(_np.asarray(levels) - 0.5)))
    return 10  # фолбэк: середина 21 уровня


def get_toto2():
    """Синглтон Toto2Model (лениво; toto2/torch импортируются тут)."""
    global _toto2
    if _toto2 is not None:
        return _toto2
    from toto2 import Toto2Model

    _set_threads()
    model = Toto2Model.from_pretrained(TOTO2_REPO)
    model.eval()
    _toto2 = model
    return _toto2


def toto2_batch(inputs: list[np.ndarray], horizon: int, batch_size: int = 8) -> np.ndarray:
    """Point-прогноз Toto 2.0 (медиана q50): (B, H).

    model.forecast() -> (9, B, n_variates, H), уровни 0.1..0.9 -> медиана idx 4.
    Контекст обрезается до последних TOTO_CTX (как у TimesFM), затем длина
    округляется ВНИЗ до кратного patch=32 (минус <=31 самая старая точка —
    патч-токенизатор требует делимость; на пары/метрики не влияет).
    Батчи группируются по одинаковой длине (raw-API требует прямоугольный тензор).
    """
    import torch

    model = get_toto2()
    groups: dict[int, list[tuple[int, np.ndarray]]] = {}
    for i, x in enumerate(inputs):
        a = np.asarray(x, dtype=float)[-TOTO_CTX:]
        a = a[-(len(a) // 32 * 32):]          # делимость на patch 32
        groups.setdefault(len(a), []).append((i, a))

    out = np.zeros((len(inputs), horizon))
    for _L, items in groups.items():
        for j in range(0, len(items), batch_size):
            chunk = items[j : j + batch_size]
            target = torch.tensor(np.stack([a for _, a in chunk]), dtype=torch.float32).unsqueeze(1)
            mask = torch.ones_like(target, dtype=torch.bool)
            series_ids = torch.zeros(len(chunk), 1, dtype=torch.long)
            with torch.no_grad():
                q = model.forecast(
                    {"target": target, "target_mask": mask, "series_ids": series_ids},
                    horizon=horizon,
                    decode_block_size=TOTO_DECODE_BLOCK,
                    has_missing_values=False,
                )
            med = q[:, :, 0, :].detach().cpu().numpy()[4]  # (B, H)
            for k, (i, _) in enumerate(chunk):
                out[i] = med[k]
    if np.isnan(out).any():
        raise ValueError("toto2: NaN в прогнозах")
    return out


def extra_forecast_batch(name: str, inputs: list[np.ndarray], horizon: int) -> np.ndarray:
    """Диспетчер по имени: 'chronos2' | 'toto4m'."""
    if name == "chronos2":
        return chronos2_batch(inputs, horizon)
    if name == "toto4m":
        return toto2_batch(inputs, horizon)
    raise KeyError(f"неизвестная extra-модель: {name}")
