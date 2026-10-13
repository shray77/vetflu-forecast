import numpy as np
import pytest

from vetflu_forecast.config import HORIZON_M4
from vetflu_forecast.data_m4 import load_m4_weekly


@pytest.fixture(scope="module")
def m4_items():
    return load_m4_weekly()


def test_loader_returns_sample(m4_items):
    assert len(m4_items) == 120
    ids = [it["id"] for it in m4_items]
    assert len(set(ids)) == 120


def test_lengths_and_horizon(m4_items):
    for it in m4_items[:20]:
        assert len(it["train"]) >= 117
        assert it["test"] is not None and len(it["test"]) == HORIZON_M4


def test_deterministic_sample():
    a = load_m4_weekly(n_series=30)
    b = load_m4_weekly(n_series=30)
    assert [it["id"] for it in a] == [it["id"] for it in b]


def test_values_numeric(m4_items):
    it = m4_items[0]
    assert np.issubdtype(it["train"].dtype, np.floating)
    assert np.all(np.isfinite(it["train"]))
