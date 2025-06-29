import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Подавление предупреждений
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Загрузка данных
training_df = pd.read_csv('training.csv', sep=';', decimal=',')
check_df = pd.read_csv('check.csv', sep=';', decimal=',')

# Обработка данных
def preprocess_data(df1):
    df1["period"] = pd.to_datetime(df1["period"], format="%d.%m.%Y")
    df1["month"] = df1["period"].dt.month
    df1["quarter"] = df1["period"].dt.quarter
    df1["dayofweek"] = df1["period"].dt.dayofweek
    df1["is_weekend"] = (df1["dayofweek"] >= 5).astype(int)
    df1["month_sin"] = np.sin(2 * np.pi * df1["month"] / 12)
    df1["month_cos"] = np.cos(2 * np.pi * df1["month"] / 12)
    df1.drop(["period"], axis=1, inplace=True)
    return df1

training_df = preprocess_data(training_df)
check_df = preprocess_data(check_df)
training_df["living_persons"] = np.log1p(training_df["living_persons"])

# Разделение данных
X, y = training_df.drop(['living_persons'], axis=1), training_df['living_persons']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели с лучшими параметрами
best_params = {
    'iterations': 1917,
    'depth': 7,
    'learning_rate': 0.08211287791566783,
    'l2_leaf_reg': 1.6965713888281966,
    'bagging_temperature': 4.654890803227147,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 100
}

final_model = CatBoostRegressor(**best_params)
final_model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = final_model.predict(X_test)
rmse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred), squared=False)
print(f"Root Mean Squared Error на тесте: {rmse:.4f}")

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Предсказания на контрольной выборке
check_ids = check_df['flat_id']  # Сохраняем идентификаторы
check_features = check_df.drop(['flat_id'], axis=1)  # Убираем flat_id

# Убедитесь, что в check_features есть те же признаки, что и в X_train
check_features = check_features.reindex(columns=X.columns, fill_value=0)

check_predictions = np.expm1(final_model.predict(check_features)).round().astype(int)

# Создание файла результата
result_df = pd.DataFrame({'flat_id': check_ids, 'living_persons': check_predictions})
result_df.to_csv('result_optuna_giperparam_catboost.csv', index=False)

print("Результаты сохранены в файл result_optuna_giperparam_catboost.csv")