"""Ретроспектива (бэктест): rolling-origin сравнение TimesFM против бейзлайнов.

Вопрос, на который отвечает модуль: даёт ли TimesFM 2.5 реальную пользу на наших
недельных рядах (ВПГП по флайвеям, M4 Weekly) против дешёвой статистики?
Масштаб честный — сезонный MASE (m=52), как в литературе по M4.
"""
from __future__ import annotations

import numpy as np

from .baselines import get_baselines
from .config import MIN_CTX, PER_CORE_BATCH_SIZE, SEASON, VERDICT_MARGIN
from .metrics import mase, smape, wape


def rolling_origins(L: int, h: int, n_origins: int, step: int, min_ctx: int = MIN_CTX) -> list[int]:
    """Индексы-концы контекста: ctx = y[:end], actual = y[end:end+h]."""
    ends = []
    for k in range(n_origins):
        end = L - k * step - h
        if end < min_ctx:
            break
        ends.append(end)
    return sorted(ends)


def season_strength(ctx: np.ndarray, m: int = SEASON) -> float:
    """Нормированная автокорреляция на лаге m по хвосту ряда (сила сезонности).

    Хвост ~3 сезона: для чистого синуса оценка стремится к (1 - m/N) и не липнет
    к граничному порогу 0.5, как на ровно 2 сезонах.
    """
    x = np.asarray(ctx, dtype=float)
    if len(x) <= m:
        return 0.0
    tail = x[-(3 * m):] if len(x) > 3 * m else x
    x = tail - tail.mean()
    denom = float((x * x).sum())
    if denom < 1e-12:
        return 0.0
    return float((x[m:] * x[:-m]).sum() / denom)


def _bucket(strength: float) -> str:
    if strength >= 0.5:
        return "strong"
    if strength >= 0.2:
        return "mid"
    return "weak"


def build_pairs(items: list[dict], h: int, n_origins: int, step: int, min_ctx: int = MIN_CTX) -> list[dict]:
    """items: {'key', 'y', 'future'?}. future — фактический хвост вне y (holdout M4)."""
    pairs = []
    for it in items:
        y = np.asarray(it["y"], dtype=float)
        future = it.get("future")
        n_roll = n_origins - (1 if future is not None and len(future) >= h else 0)
        for end in rolling_origins(len(y), h, n_roll, step, min_ctx):
            pairs.append({"key": it["key"], "ctx": y[:end], "actual": y[end:end + h],
                          "bucket": _bucket(season_strength(y[:end]))})
        if future is not None and len(future) >= h:
            pairs.append({"key": it["key"], "ctx": y, "actual": np.asarray(future[:h], dtype=float),
                          "bucket": _bucket(season_strength(y))})
    return pairs


def make_verdict(mase_tfm: float, best_name: str, mase_best: float,
                 margin: float = VERDICT_MARGIN) -> dict:
    """Скилл относительно лучшего бейзлайна: >+5% польза, ±5% паритет, иначе избыточен."""
    skill = 1.0 - mase_tfm / mase_best if mase_best > 0 else 0.0
    if skill > margin:
        status = "value"
        text = f"TimesFM бьёт лучший бейзлайн ({best_name}) на {skill*100:.1f}% по MASE — пользу даёт"
    elif skill >= -margin:
        status = "parity"
        text = f"Паритет с лучшим бейзлайном ({best_name}) — TimesFM не хуже, но и не лучше"
    else:
        status = "no_value"
        text = f"Лучший бейзлайн ({best_name}) бьёт TimesFM на {-skill*100:.1f}% по MASE — foundation-модель тут избыточна"
    return {"skill": round(skill, 4), "status": status, "best_baseline": best_name, "text": text}


def _aggregate(pairs, model_names, preds: dict[str, np.ndarray]) -> list[dict]:
    per_model = {mn: {"mase": [], "wape": [], "smape": [], "se": [], "ae": [], "ay": []}
                 for mn in model_names}
    for i, pair in enumerate(pairs):
        for mn in model_names:
            pred = preds[mn][i]
            per_model[mn]["mase"].append(mase(pair["actual"], pred, pair["ctx"]))
            per_model[mn]["wape"].append(wape(pair["actual"], pred))
            per_model[mn]["smape"].append(smape(pair["actual"], pred))
            per_model[mn]["se"].extend(((pair["actual"] - pred) ** 2).tolist())
            per_model[mn]["ae"].append(float(np.abs(pair["actual"] - pred).sum()))
            per_model[mn]["ay"].append(float(np.abs(pair["actual"]).sum()))
    summary = []
    for mn in model_names:
        d = per_model[mn]
        summary.append({
            "model": mn,
            "mase": round(float(np.mean(d["mase"])), 4),
            "wape": round(float(np.sum(d["ae"]) / max(np.sum(d["ay"]), 1e-12)), 4),
            "smape": round(float(np.mean(d["smape"])), 2),
            "rmse": round(float(np.sqrt(np.mean(d["se"]))), 4),
            "n_pairs": len(d["mase"]),
        })
    return summary


def run_backtest(items: list[dict], h: int, n_origins: int, step: int,
                 with_hw: bool = False, use_timesfm: bool = True, label: str = "") -> dict:
    """Полный цикл: пары (серия, origin) -> прогнозы всех моделей -> агрегаты + вердикт."""
    baselines = get_baselines(with_hw=with_hw)
    model_names = list(baselines.keys()) + (["timesfm"] if use_timesfm else [])
    pairs = build_pairs(items, h, n_origins, step)
    if not pairs:
        raise ValueError(f"[{label}] не набралось ни одной пары (короткие ряды?)")

    preds: dict[str, np.ndarray] = {mn: np.zeros((len(pairs), h)) for mn in model_names}
    ctxs = [p["ctx"] for p in pairs]
    for mn, fn in baselines.items():
        preds[mn] = np.asarray([fn(c, h) for c in ctxs], dtype=float)

    if use_timesfm:
        from .model import forecast_batch  # лениво: torch грузится только тут

        flat = []
        for i in range(0, len(ctxs), PER_CORE_BATCH_SIZE):
            chunk = ctxs[i:i + PER_CORE_BATCH_SIZE]
            p, _ = forecast_batch(chunk, h)
            flat.append(p)
        preds["timesfm"] = np.concatenate(flat, axis=0)

    summary = _aggregate(pairs, model_names, preds)

    # разрезы по силе сезонности
    buckets = []
    for b in ("strong", "mid", "weak"):
        idx = [i for i, p in enumerate(pairs) if p["bucket"] == b]
        if not idx:
            continue
        sub_preds = {mn: preds[mn][idx] for mn in model_names}
        sub_pairs = [pairs[i] for i in idx]
        bsum = _aggregate(sub_pairs, model_names, sub_preds)
        buckets.append({"bucket": b, "n_pairs": len(idx), "summary": bsum})

    base_rows = [r for r in summary if r["model"] != "timesfm"]
    best = min(base_rows, key=lambda r: r["mase"])
    if use_timesfm:
        tfm = next(r for r in summary if r["model"] == "timesfm")
        verdict = make_verdict(tfm["mase"], best["model"], best["mase"])
    else:
        verdict = None

    return {
        "label": label,
        "h": h,
        "n_series": len(items),
        "n_pairs": len(pairs),
        "summary": summary,
        "buckets": buckets,
        "verdict": verdict,
    }
