import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

training_df = pd.read_csv('training.csv', sep=';', decimal=',', date_format='DD.MM.YYYY')
check_df = pd.read_csv('check.csv', sep=';', decimal=',', date_format='DD.MM.YYYY')
print(training_df.shape)
print(training_df.head())
print(check_df.shape)
print(check_df.head())

# Конвертация столбца `period` в дату и создание новых признаков
training_df["period"] = pd.to_datetime(training_df["period"], format="%d.%m.%Y")
training_df["year"] = training_df["period"].dt.year
training_df["month"] = training_df["period"].dt.month
training_df["day"] = training_df["period"].dt.day
training_df.drop(["period"], axis=1, inplace=True)

# Разделение данных
X, y = training_df.drop(['living_persons'], axis=1), training_df['living_persons']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

# Обучение модели
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# Предсказания и метрика
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse:.4f}")

# Проверка важности признаков
feature_importances = model.get_feature_importance(prettified=True)
print(feature_importances)

check_df.drop(["period"], axis=1, inplace=True)

# Предсказания на контрольной выборке
check_ids = check_df['flat_id']  # Сохраняем идентификаторы
check_features = check_df.drop(['flat_id'], axis=1)  # Убираем flat_id

# Убедитесь, что в check_features есть те же признаки, что и в X_train
check_features = check_features.reindex(columns=X.columns, fill_value=0)

check_predictions = np.expm1(model.predict(check_features)).round().astype(int)

# Создание файла результата
result_df = pd.DataFrame({'flat_id': check_ids, 'living_persons': check_predictions})
result_df.to_csv('result_base_line.csv', index=False)

print("Результаты сохранены в файл result_base_line.csv")