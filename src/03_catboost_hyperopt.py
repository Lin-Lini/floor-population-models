import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Загрузка данных
training_df = pd.read_csv('training.csv', sep=';', decimal=',')
check_df = pd.read_csv('check.csv', sep=';', decimal=',')

# Обработка данных
training_df["period"] = pd.to_datetime(training_df["period"], format="%d.%m.%Y")
training_df["month"] = training_df["period"].dt.month
training_df["quarter"] = training_df["period"].dt.quarter
training_df["dayofweek"] = training_df["period"].dt.dayofweek
training_df["is_weekend"] = (training_df["dayofweek"] >= 5).astype(int)
training_df["month_sin"] = np.sin(2 * np.pi * training_df["month"] / 12)
training_df["month_cos"] = np.cos(2 * np.pi * training_df["month"] / 12)
training_df["living_persons"] = np.log1p(training_df["living_persons"])
training_df.drop(["period"], axis=1, inplace=True)

check_df["period"] = pd.to_datetime(check_df["period"], format="%d.%m.%Y")
check_df["month"] = check_df["period"].dt.month
check_df["quarter"] = check_df["period"].dt.quarter
check_df["dayofweek"] = check_df["period"].dt.dayofweek
check_df["is_weekend"] = (check_df["dayofweek"] >= 5).astype(int)
check_df["month_sin"] = np.sin(2 * np.pi * check_df["month"] / 12)
check_df["month_cos"] = np.cos(2 * np.pi * check_df["month"] / 12)
check_df.drop(["period"], axis=1, inplace=True)

# Разделение данных
X, y = training_df.drop(['living_persons'], axis=1), training_df['living_persons']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Функция для Optuna
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0, 5),
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": 0
    }

    # Кросс-валидация
    train_pool = Pool(X, y)
    cv_results = cv(
        params=params,
        pool=train_pool,
        fold_count=5,
        shuffle=True,
        partition_random_seed=42,
        verbose=False
    )

    # Минимальный RMSE из кросс-валидации
    return cv_results['test-RMSE-mean'].min()

# Запуск Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Лучшие параметры
best_params = study.best_params
print("Лучшие параметры:", best_params)

# Обучение модели с лучшими параметрами
final_model = CatBoostRegressor(**best_params)
final_model.fit(X, y, verbose=100)

# Предсказания на тестовой выборке
y_pred = final_model.predict(X_test)
rmse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred), squared=False)
print(f"Root Mean Squared Error на тесте: {rmse:.4f}")