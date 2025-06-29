# 🏠 Volga IT 2024 — Прогноз численности проживающих

Репозиторий содержит **10 решений задачи регрессии/классификации численности квартир** на основе данных `training.csv` и `check.csv`.

## 📁 Структура проекта

```
volga_it_2024_population_models/
├── data/                           # CSV-файлы (training, check)
├── img/                            # Визуализации и метрики
├── src/                            # 10 скриптов с разными моделями
├── .gitignore
├── README.md
└── requirements.txt
```

## 🔬 Варианты решений

| № | Файл в `src/` | Модель/подход |
|--:|------------------------------|-------------------------------------------|
| 01 | 01_baseline_catboost.py | CatBoostRegressor, базовый |
| 02 | 02_baseline_catboost_extra_features.py | CatBoost + новые признаки |
| 03 | 03_catboost_hyperopt.py | Подбор гиперпараметров CatBoost |
| 04 | 04_catboost_tuned.py | CatBoost с оптимизированными параметрами |
| 05 | 05_random_forest_regressor.py | RandomForestRegressor |
| 06 | 06_rf_classifier_gridsearch.py | RandomForestClassifier + GridSearchCV |
| 07 | 07_catboost_classifier.py | CatBoostClassifier |
| 08 | 08_catboost_regression_filtered.py | CatBoost с фильтрацией (1–7 проживающих) |
| 09 | 09_combined_catboost_rf.py | CatBoost + RF комбинированный подход |
| 10 | 10_final_classification_solution.py | Финальное: CatBoost + RF классификация |

## 🚀 Быстрый старт

```bash
pip install -r requirements.txt
python src/01_baseline_catboost.py
```

## 📌 Зависимости

Список в `requirements.txt`. Используются:
- pandas, numpy
- scikit-learn
- catboost
- matplotlib, seaborn
- optuna (для гиперпараметров)
- imbalanced-learn

## 📄 Лицензия

MIT
