"""A/B: Chronos-2 / Toto-2.0-4m против TimesFM 2.5 и бейзлайнов на тех же панелях.

Запускается ПО ОДНОЙ модели за процесс (стеснённая RAM: адаптеры синглтоны,
но лучше каждый вызов = новый процесс):

  python scripts/ab_run.py --panel all --model baselines
  python scripts/ab_run.py --panel wahis --model toto4m
  python scripts/ab_run.py --panel regions --model chronos2
  python scripts/ab_run.py --panel meteo --model timesfm   # эксперимент «погода через TimesFM»

Результаты: data/ab/{panel}__{model}.json (summary-строки в формате _aggregate).
Сводка: python scripts/ab_merge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vetflu_forecast import config  # noqa: E402
from vetflu_forecast.backtest import _aggregate, build_pairs, get_baselines  # noqa: E402

OUT_DIR = config.DATA_DIR / "ab"

PANELS = {
    "m4": {"h": config.HORIZON_M4, "n_origins": config.M4_ORIGINS, "step": config.M4_ORIGIN_STEP},
    "wahis": {"h": config.HORIZON_WAHIS, "n_origins": config.WAHIS_ORIGINS, "step": config.WAHIS_ORIGIN_STEP},
    "regions": {"h": config.HORIZON_REGIONS, "n_origins": config.REGIONS_ORIGINS, "step": config.REGIONS_ORIGIN_STEP},
    "meteo": {"h": config.HORIZON_WAHIS, "n_origins": 6, "step": 8},
}


def panel_items(panel: str, force_meteo: bool = False) -> list[dict]:
    if panel == "m4":
        from vetflu_forecast.data_m4 import load_m4_weekly

        return [{"key": it["id"], "y": it["train"], "future": it["test"]}
                for it in load_m4_weekly()]
    if panel == "wahis":
        from vetflu_forecast.pipeline import _get_wahis_series

        return [{"key": k, "y": s.to_numpy(dtype=float)}
                for k, s in sorted(_get_wahis_series().items())]
    if panel == "regions":
        from vetflu_forecast.data_regions import region_series
        from vetflu_forecast.pipeline import _get_outbreaks

        region_map, _meta, _districts = region_series(_get_outbreaks())
        return [{"key": k, "y": v["smooth"].to_numpy(dtype=float)}
                for k, v in sorted(region_map.items())]
    if panel == "meteo":
        from vetflu_forecast.data_meteo import load_meteo_items

        return load_meteo_items(force=force_meteo)
    raise KeyError(panel)


def run(panel: str, model: str, force_meteo: bool = False) -> dict:
    cfg = PANELS[panel]
    items = panel_items(panel, force_meteo=force_meteo)
    pairs = build_pairs(items, cfg["h"], cfg["n_origins"], cfg["step"])
    print(f"[{panel}] серий: {len(items)}, пар: {len(pairs)}, h={cfg['h']}", flush=True)

    if model == "baselines":
        baselines = get_baselines(with_hw=config.USE_HOLT_WINTERS.get(panel, False))
        preds = {mn: np.asarray([fn(p["ctx"], cfg["h"]) for p in pairs], dtype=float)
                 for mn, fn in baselines.items()}
        summary = _aggregate(pairs, list(baselines.keys()), preds)
        name = "baselines"
    else:
        if model == "timesfm":
            from vetflu_forecast.model import forecast_batch

            ctxs = [p["ctx"] for p in pairs]
            flat = []
            B = config.PER_CORE_BATCH_SIZE
            for i in range(0, len(ctxs), B):
                p, _ = forecast_batch(ctxs[i:i + B], cfg["h"])
                flat.append(p)
            preds = np.concatenate(flat, axis=0)
        else:
            from vetflu_forecast.models_extra import extra_forecast_batch

            preds = extra_forecast_batch(model, [p["ctx"] for p in pairs], cfg["h"])
        summary = _aggregate(pairs, [model], {model: preds})
        name = model

    doc = {
        "panel": panel, "model": name, "h": cfg["h"],
        "n_series": len(items), "n_pairs": len(pairs),
        "summary": summary,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{panel}__{name}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    for row in summary:
        print(f"  {row['model']:10s} MASE={row['mase']:.4f} WAPE={row['wape']:.4f} "
              f"sMAPE={row['smape']:.2f} RMSE={row['rmse']:.4f}", flush=True)
    print(f"  -> {path}", flush=True)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True,
                    choices=["m4", "wahis", "regions", "meteo", "all"])
    ap.add_argument("--model", required=True,
                    choices=["baselines", "chronos2", "toto4m", "timesfm"])
    ap.add_argument("--force-meteo", action="store_true",
                    help="пересобрать кэш метео (иначе data/meteo_cache.json)")
    args = ap.parse_args()

    panels = list(PANELS) if args.panel == "all" else [args.panel]
    for p in panels:
        run(p, args.model, force_meteo=args.force_meteo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
