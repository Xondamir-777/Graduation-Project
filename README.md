# AgroMonitor UZ — Мобильное приложение + Backend

## Структура

```
agromonitor/
├── backend/
│   ├── main.py           ← FastAPI backend
│   └── requirements.txt
└── mobile_app/
    └── index.html        ← PWA мобильное приложение
```

---

## Запуск Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend автоматически ищет модель в `../../plant_disease_project/`.
Если модель не найдена — работает в demo-режиме (случайные предсказания).

### API Endpoints

| Method | URL | Описание |
|---|---|---|
| POST | `/predict` | Предсказание болезни |
| POST | `/feedback/{id}` | Отзыв агронома |
| GET | `/observations` | Список наблюдений |
| GET | `/dataset/export` | Экспорт CSV датасета |
| GET | `/stats` | Статистика |
| GET | `/health` | Проверка состояния |

---

## Запуск мобильного приложения

### Вариант 1 — Прямо в браузере
```bash
cd mobile_app
python -m http.server 3000
# Открыть http://localhost:3000
```

### Вариант 2 — На телефоне (через локальную сеть)
```bash
# Backend на компьютере:
uvicorn main:app --host 0.0.0.0 --port 8000

# В приложении → Настройки → URL:
# http://192.168.1.XXX:8000   (IP вашего компьютера)
```

### Вариант 3 — Добавить на домашний экран (PWA)
1. Открыть в Chrome/Safari на телефоне
2. Поделиться → Добавить на экран «Домой»
3. Приложение работает как нативное

---

## Как работает сбор реального датасета

```
Агроном фотографирует лист
        ↓
Приложение собирает параметры:
  - Регион, сезон
  - Температура, влажность, осадки
  - Тип почвы, pH
        ↓
POST /predict → модель предсказывает болезнь
        ↓
Наблюдение сохраняется в SQLite (agromonitor.db)
        ↓
Агроном может оставить отзыв:
  POST /feedback/{id} → true_class, notes
        ↓
GET /dataset/export → CSV для дообучения модели
```

---

## Схема БД (SQLite)

| Поле | Тип | Описание |
|---|---|---|
| id | TEXT | UUID наблюдения |
| created_at | DATETIME | Время |
| region | TEXT | Регион Узбекистана |
| season | TEXT | Сезон |
| avg_temperature_c | FLOAT | Температура °C |
| humidity_pct | FLOAT | Влажность % |
| rainfall_mm | FLOAT | Осадки мм |
| soil_type | TEXT | Тип почвы |
| soil_ph | FLOAT | pH почвы |
| image_path | TEXT | Путь к сохранённому фото |
| predicted_class | TEXT | Предсказанная болезнь |
| disease_probability | FLOAT | Вероятность [0,1] |
| has_feedback | BOOL | Есть ли отзыв |
| true_class | TEXT | Реальная болезнь (от агронома) |
| feedback_notes | TEXT | Заметки агронома |
| agronomist_name | TEXT | Имя агронома |

---

## Экспорт датасета для дообучения

```bash
# Все наблюдения
curl http://localhost:8000/dataset/export > real_dataset.csv

# Только верифицированные (с отзывами агрономов)
curl "http://localhost:8000/dataset/export?only_with_feedback=true" > verified_dataset.csv
```

Затем объедини с синтетическим датасетом:
```python
import pandas as pd
synthetic = pd.read_csv("data/dataset.csv")
real      = pd.read_csv("real_dataset.csv")
combined  = pd.concat([synthetic, real], ignore_index=True)
combined.to_csv("data/combined_dataset.csv", index=False)
```
