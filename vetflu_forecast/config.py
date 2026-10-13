"""Конфиг vetflu-forecast: модель, пути, горизонты, WAHIS, бэктест.

Всё, что меняется при развитии пилота, собрано здесь. Ревизия модели запинена,
чтобы ключ actions/cache был стабильным (см. README «Кэш модели»).
Env-оверрайды VTFC_CTX / VTFC_BATCH — для стеснённых машин (маленькая RAM).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SITE_DIR = ROOT / "site"
SITE_DATA_DIR = SITE_DIR / "data"

# --- модель (код и веса 2.5 — Apache-2.0; 3.0 non-commercial, НЕ используем) ---
MODEL_REPO = "google/timesfm-2.5-200m-pytorch"
MODEL_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"  # пин ревизии для стабильного кэша
MAX_CONTEXT = int(os.environ.get("VTFC_CTX", "512"))   # вход обрезается/паддится (512 нед ~ 10 лет)
MAX_HORIZON = 16         # compile-time максимум (M4 h=13 и WAHIS h=4 укладываются)
PER_CORE_BATCH_SIZE = int(os.environ.get("VTFC_BATCH", "8"))  # CPU device_count=1 -> батч = это число

# --- ряды ---
SEASON = 52              # недельная сезонность
HORIZON_WAHIS = 4        # «прогноз +4 недели» — конвенция витрины vetflu-site
HORIZON_M4 = 13          # стандартный горизонт M4 Weekly
MIN_CTX = 2 * SEASON + 1  # минимум контекста у origin (2 полных года)

# --- WAHIS (публичный POST API, как в vetflu-site/scripts/fetch_outbreaks.mjs) ---
WAHIS_API = "https://wahis.woah.org/api/v1/pi/event/filtered-list?language=en"
WAHIS_PAGE_SIZE = 500
WAHIS_MAX_PAGES = 60
WAHIS_SLEEP = 0.4

# Коридоры флайвеев — синхронизировано с vetflu-site (primary = полный вес, adjacent = 0.4).
FLYWAYS = ["ea", "bsm", "ca", "eaa"]
FLYWAY_LABELS = {
    "global": "ВПГП, весь мир",
    "ea": "Восточно-Атлантический",
    "bsm": "Черноморско-Средиземноморский",
    "ca": "Центральноазиатский",
    "eaa": "Восточноазиатско-Австралазийский",
}
CORRIDORS = {
    "ea": {
        "primary": ["russ", "finland", "estonia", "latvia", "lithuania", "poland", "belarus",
                    "germany", "denmark", "netherlands", "norway", "sweden"],
        "adjacent": ["ukraine", "moldova", "romania", "hungary", "slovakia", "czech",
                     "united kingdom", "iceland", "ireland", "france", "belgium"],
    },
    "bsm": {
        "primary": ["russ", "ukraine", "moldova", "romania", "bulgaria", "turkey", "türkiye",
                    "turkiye", "georgia", "armenia", "azerbaijan", "kazakhstan"],
        "adjacent": ["hungary", "serbia", "greece", "iran", "iraq", "syria", "israel",
                     "lebanon", "jordan", "turkmenistan", "uzbekistan", "slovakia"],
    },
    "ca": {
        "primary": ["russ", "kazakhstan", "uzbekistan", "turkmenistan", "kyrgyzstan", "tajikistan"],
        "adjacent": ["mongolia", "iran", "azerbaijan", "georgia", "afghanistan", "china",
                     "turkey", "türkiye", "turkiye", "ukraine"],
    },
    "eaa": {
        "primary": ["russ", "mongolia", "china", "japan", "korea"],
        "adjacent": ["vietnam", "kazakhstan", "india", "nepal", "thailand", "myanmar"],
    },
}

# --- vet-heatmap: регионы РФ (округа × группы болезней) ---
# Источник: витрина vet-heatmap, data/outbreaks.json (обновляется каждые 6 ч,
# CORS *). Ряды сглаживаются скользящим средним: на сырых счётчиках TimesFM в
# паритете с наивной (панель wahis), на сглаженных ожидаем пользу.
HEATMAP_OUTBREAKS_URL = "https://shray77.github.io/vet-heatmap/data/outbreaks.json"
HORIZON_REGIONS = 4
REGIONS_START = "2023-01-01"   # плотная эра данных начинается в 2024; даём год контекста
REGIONS_SMOOTH_WEEKS = 4       # окно скользящего среднего (недель)
REGIONS_MIN_EVENTS = 24        # минимум событий с REGIONS_START, иначе серия не берём
REGIONS_ORIGINS = 6
REGIONS_ORIGIN_STEP = 8

# --- бэктест ---
BACKTEST_SEED = 42
M4_N_SERIES = 120        # выборка из 359 рядов M4 Weekly (стратификация по длине не нужна)
M4_ORIGINS = 4
M4_ORIGIN_STEP = 13
M4_MIN_LEN = MIN_CTX + HORIZON_M4          # 117
WAHIS_ORIGINS = 6
WAHIS_ORIGIN_STEP = 8
USE_HOLT_WINTERS = {"wahis": True, "m4": False, "regions": True}  # HW дорог: только на своих рядах

VERDICT_MARGIN = 0.05  # |skill| <= 5% -> паритет
