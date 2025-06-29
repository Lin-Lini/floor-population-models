import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings

training_df = pd.read_csv('training.csv', sep=';', decimal=',', date_format='DD.MM.YYYY')

warnings.filterwarnings('ignore', category=FutureWarning)

training_df["period"] = pd.to_datetime(training_df["period"], format="%d.%m.%Y")
training_df["month"] = training_df["period"].dt.month
training_df["quarter"] = training_df["period"].dt.quarter
training_df["dayofweek"] = training_df["period"].dt.dayofweek
training_df["is_weekend"] = (training_df["dayofweek"] >= 5).astype(int)
training_df["month_sin"] = np.sin(2 * np.pi * training_df["month"] / 12)
training_df["month_cos"] = np.cos(2 * np.pi * training_df["month"] / 12)
training_df["living_persons"] = np.log1p(training_df["living_persons"])

training_df.drop(["period"], axis=1, inplace=True)

X, y = training_df.drop(['living_persons'], axis=1), training_df['living_persons']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.01,
    l2_leaf_reg=8,
    bagging_temperature=3.0,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

# Обучение модели
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)
model.fit(train_pool, eval_set=test_pool, verbose=100)

# Предсказания и метрика
y_pred = model.predict(X_test)
rmse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred), squared=False)  # Обратное преобразование
print(f"Root Mean Squared Error: {rmse:.4f}")

# Проверка важности признаков
feature_importances = model.get_feature_importance(prettified=True)
print(feature_importances)

errors = y_test - y_pred
plt.hist(errors, bins=50, alpha=0.75)
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

check_df = pd.read_csv('check.csv', sep=';', decimal=',', date_format='DD.MM.YYYY')

check_df["period"] = pd.to_datetime(check_df["period"], format="%d.%m.%Y")
check_df["month"] = check_df["period"].dt.month
check_df["quarter"] = check_df["period"].dt.quarter
check_df["dayofweek"] = check_df["period"].dt.dayofweek
check_df["is_weekend"] = (check_df["dayofweek"] >= 5).astype(int)
check_df["month_sin"] = np.sin(2 * np.pi * check_df["month"] / 12)
check_df["month_cos"] = np.cos(2 * np.pi * check_df["month"] / 12)

check_df.drop(["period"], axis=1, inplace=True)

# Предсказания на контрольной выборке
check_ids = check_df['flat_id']  # Сохраняем идентификаторы
check_features = check_df.drop(['flat_id'], axis=1)  # Убираем flat_id

# Убедитесь, что в check_features есть те же признаки, что и в X_train
check_features = check_features.reindex(columns=X.columns, fill_value=0)

check_predictions = np.expm1(model.predict(check_features)).round().astype(int)

# Создание файла результата
result_df = pd.DataFrame({'flat_id': check_ids, 'living_persons': check_predictions})
result_df.to_csv('result_base_line_upgrade.csv', index=False)

print("Результаты сохранены в файл result_base_line_upgrade.csv")