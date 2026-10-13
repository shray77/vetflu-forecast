# VetFlu Forecast — TimesFM 2.5 (пилот)

[![ci](https://github.com/shray77/vetflu-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/shray77/vetflu-forecast/actions/workflows/ci.yml)
[![forecast](https://github.com/shray77/vetflu-forecast/actions/workflows/forecast.yml/badge.svg)](https://github.com/shray77/vetflu-forecast/actions/workflows/forecast.yml)

Пилот foundation-модели **TimesFM 2.5** (200M, Apache-2.0) для экосистемы vet-*:
недельный прогноз вспышек высокопатогенного гриппа птиц (ВПГП) по 4 флайвеям + мир
(WOAH WAHIS) на GitHub Actions (CPU, без подписок) — и **ретроспектива**: даёт ли
foundation-модель реальную пользу против простой статистики на наших рядах.

**Живая витрина:** https://shray77.github.io/vetflu-forecast/

## Состав

- `forecast.yml` — воркер: понедельник 06:00 UTC + ручной dispatch.
  WAHIS → свежий снапшот → бэктест против бейзлайнов → прогноз (+4 недели) → JSON → Pages.
- `ci.yml` — быстрые тесты (модель не нужна) + smoke-тест реальной TimesFM 2.5.
- Кэш модели: `actions/cache` → `~/.cache/huggingface` (925 МБ), ключ запинен на ревизию
  чекпоинта — воркер не качает модель каждый запуск (попадание видно в step summary job'а).
- Место в экосистеме: усиление ядра [vetflu-site](https://shray77.github.io/vetflu-site/)
  (решение Task 56 ворклога: «стоит, но узко» → пилот отдельной репой).

## Результаты ретроспективы

Первый прогон воркера: run 38411269455 (2026-10-13, runner ubuntu-latest, CPU).
Модельный кэш подтверждён: `Cache restored from key: hf-ubuntu-timesfm-2.5-200m-1d95242`
— веса (925 МБ) не скачиваются повторно ни CI, ни воркером.

**M4 Weekly** (методологическая проверка; 120 рядов, h=13, 480 пар rolling-origin + holdout):

| модель | MASE ↓ | WAPE | sMAPE |
|---|---|---|---|
| **TimesFM 2.5** | **0.368** | **3.7%** | 4.5% |
| drift (лучший бейзлайн) | 0.541 | 5.1% | 7.0% |
| naive | 0.546 | 5.2% | 7.0% |
| sma8 | 0.583 | 5.8% | 8.0% |
| snaive | 1.335 | 13.2% | 13.8% |

**Свои ряды WAHIS** (ВПГП: мир + 4 флайвея; h=4 недели, 30 пар rolling-origin):

| модель | MASE ↓ | WAPE | sMAPE |
|---|---|---|---|
| naive (лучший бейзлайн) | 0.908 | 76.9% | 88.9% |
| drift | 0.910 | 77.0% | 115.6% |
| hw (Хольт-Уинтерс) | 1.174 | 86.1% | 149.7% |
| sma8 | 1.199 | 93.7% | 99.6% |
| **TimesFM 2.5** | 0.951 | 79.0% | 151.1% |
| snaive | 1.468 | 122.1% | 109.3% |

**Вердикт (честный):** на типовых недельных рядах TimesFM 2.5 уверенно бьёт статистику
(MASE −32% против лучшего бейзлайна). На сырых недельных счётчиках ВПГП — **паритет
с наивной** (−4.7%, в пределах ±5%): ряд пиковый, низкочастотный, события не повторяются
по календарю неделю-в-неделю (сезонная наивная тут вообще худшая — MASE 1.47).
Практический вывод: foundation-модель имеет смысл применять не к сырым счётчикам
событий, а к производным/сглаженным сигналам экосистемы (индекс R vetflu, заболеваемость
vet-heatmap) и коротким рядам без явной сезонной структуры — там, где наивная статистика
слабеет. Точная цифра прогноза на текущую неделю — на витрине.

## Запуск локально

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m vetflu_forecast.cli all          # бэктест + прогноз -> site/data/*.json
python -m pytest -m "not slow"             # быстрые тесты
python -m pytest -m slow                   # smoke реальной модели
```

## Данные и методика ретроспективы

- **Свои ряды**: недельные счётчики событий ВПГП из WOAH WAHIS (мир + 4 флайвея,
  веса стран как в vetflu-site: primary 1.0 / adjacent 0.4). Rolling-origin, h=4 недели.
- **Методологическая проверка**: M4 Weekly (вендорено в `data/`, выборка 120 из 359
  рядов, сид 42), h=13, включая holdout на M4-test.
- **Модели-соперники**: naive, сезонная наивная (m=52), drift, SMA(8), Хольт-Уинтерс
  (на своих рядах) против TimesFM 2.5.
- **Метрика**: сезонный MASE (масштаб по контексту), WAPE, sMAPE, RMSE.

## Лицензии и границы

Код — Apache-2.0. Веса TimesFM 2.5 — Apache-2.0 (самохостинг разрешён). TimesFM 3.0 —
non-commercial, **не используется**. Проект — пилот, не система раннего предупреждения.
