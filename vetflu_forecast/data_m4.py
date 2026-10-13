"""M4 Weekly (vendored CSV из Mcompetitions/M4-methods): методологическая ретроспектива.

359 реальных недельных рядов; в репу положены train (полные истории) и test
(13 фактических будущих значений, стандартный горизонт M4 Weekly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_DIR, M4_MIN_LEN, M4_N_SERIES, BACKTEST_SEED, HORIZON_M4

M4_TRAIN = DATA_DIR / "m4_weekly_train.csv"
M4_TEST = DATA_DIR / "m4_weekly_test.csv"


def _row_values(row: pd.Series) -> np.ndarray:
    v = pd.to_numeric(row.dropna(), errors="coerce").dropna().to_numpy(dtype=float)
    return v


def load_m4_weekly(n_series: int | None = None, min_len: int | None = None,
                   seed: int | None = None, with_test: bool = True) -> list[dict]:
    """Список {'id', 'train', 'test'}; выборка фиксированным сидом для воспроизводимости."""
    if not M4_TRAIN.exists():
        raise FileNotFoundError(f"нет {M4_TRAIN} — M4 Weekly должен быть вендорен в data/")
    n_series = n_series if n_series is not None else M4_N_SERIES
    min_len = min_len if min_len is not None else M4_MIN_LEN
    seed = seed if seed is not None else BACKTEST_SEED

    train = pd.read_csv(M4_TRAIN, index_col=0)
    test = pd.read_csv(M4_TEST, index_col=0) if (with_test and M4_TEST.exists()) else None

    items = []
    for sid, row in train.iterrows():
        y = _row_values(row)
        if len(y) >= min_len:
            items.append({"id": str(sid), "train": y, "test": None})
    rng = np.random.default_rng(seed)
    if 0 < n_series < len(items):
        pick = rng.choice(len(items), size=n_series, replace=False)
        items = [items[i] for i in sorted(pick)]
    if test is not None:
        for it in items:
            if it["id"] in test.index:
                it["test"] = _row_values(test.loc[it["id"]])
    return items
