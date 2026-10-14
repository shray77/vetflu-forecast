"""Сводка A/B: склеивает data/ab/{panel}__{model}.json (+ published TimesFM
из site/data/retrospective.json, если вендорен) в одну таблицу по панелям.

  python scripts/ab_merge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vetflu_forecast import config  # noqa: E402

AB = config.DATA_DIR / "ab"
PANEL_TITLES = {
    "m4": "M4 Weekly (h=13, методология)",
    "wahis": "WAHIS HPAI, сырые события (h=4)",
    "regions": "Регионы РФ, сглаженные (h=4)",
    "meteo": "Погода: недельные t/THI флайвеев (h=4) — эксперимент",
}


def published_timesfm() -> dict[str, dict[str, float]]:
    """{panel: {model: mase}} из site/data/retrospective.json (если вендорен)."""
    out = {}
    retro = config.SITE_DATA_DIR / "retrospective.json"
    if not retro.exists():
        return out
    doc = json.loads(retro.read_text())
    for p in doc.get("panels", []):
        rows = {r["model"]: r["mase"] for r in p.get("summary", [])}
        out[p["label"]] = rows
    return out


def main() -> int:
    pub = published_timesfm()
    files = sorted(AB.glob("*__*.json"))
    if not files:
        print("нет результатов в data/ab/ — сначала запусти scripts/ab_run.py")
        return 1

    per_panel: dict[str, dict[str, dict]] = {}
    for f in files:
        doc = json.loads(f.read_text())
        panel = doc["panel"]
        for row in doc["summary"]:
            per_panel.setdefault(panel, {})[row["model"]] = row

    for panel in ("m4", "wahis", "regions", "meteo"):
        if panel not in per_panel:
            continue
        rows = per_panel[panel]
        print(f"\n=== {PANEL_TITLES.get(panel, panel)} ===")
        print(f"{'модель':12s} {'MASE':>8s} {'WAPE':>8s} {'sMAPE':>8s} {'пар':>5s}")

        if panel in pub:
            for m, mase in pub[panel].items():
                print(f"{m + ' *':12s} {mase:8.4f} {'—':>8s} {'—':>8s} {'—':>5s}   (* published, воркер)")

        best = min(r["mase"] for r in rows.values())
        for name, r in rows.items():
            mark = "  <-- лучший" if abs(r["mase"] - best) < 1e-9 else ""
            print(f"{name:12s} {r['mase']:8.4f} {r['wape']:8.4f} {r['smape']:8.2f} "
                  f"{r['n_pairs']:5d}{mark}")

        baselines_file = AB / f"{panel}__baselines.json"
        if baselines_file.exists():
            bdoc = json.loads(baselines_file.read_text())
            bb = min(bdoc["summary"], key=lambda r: r["mase"])
            for name, r in rows.items():
                if name == "baselines":
                    continue
                skill = 1 - r["mase"] / bb["mase"]
                print(f"  skill {name:12s} vs {bb['model']:8s}: {skill * 100:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
