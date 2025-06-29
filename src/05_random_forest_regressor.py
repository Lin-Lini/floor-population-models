import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Указываем правильные разделители и десятичный разделитель
training_df = pd.read_csv('training.csv', sep=';', decimal=',', parse_dates=['period'], dayfirst=True)
check_df = pd.read_csv('check.csv', sep=';', decimal=',', parse_dates=['period'], dayfirst=True)

# Проверяем на наличие пропущенных значений
print(training_df.isnull().sum())
print(check_df.isnull().sum())

# Обработка данных
training_df["period"] = pd.to_datetime(training_df["period"], format="%d.%m.%Y")
training_df["month"] = training_df["period"].dt.month
training_df["quarter"] = training_df["period"].dt.quarter
training_df["dayofweek"] = training_df["period"].dt.dayofweek
training_df["is_weekend"] = (training_df["dayofweek"] >= 5).astype(int)
training_df["month_sin"] = np.sin(2 * np.pi * training_df["month"] / 12)
training_df["month_cos"] = np.cos(2 * np.pi * training_df["month"] / 12)
training_df["living_persons"] = np.log1p(training_df["living_persons"])
training_df.drop(["period", "flat_id"], axis=1, inplace=True)
training_df['sewage'] = training_df['sewage'].fillna(training_df['hot_water'] + training_df['cold_water'])

check_df['sewage'] = check_df['sewage'].fillna(check_df['hot_water'] + check_df['cold_water'])
check_df["period"] = pd.to_datetime(check_df["period"], format="%d.%m.%Y")
check_df["month"] = check_df["period"].dt.month
check_df["quarter"] = check_df["period"].dt.quarter
check_df["dayofweek"] = check_df["period"].dt.dayofweek
check_df["is_weekend"] = (check_df["dayofweek"] >= 5).astype(int)
check_df["month_sin"] = np.sin(2 * np.pi * check_df["month"] / 12)
check_df["month_cos"] = np.cos(2 * np.pi * check_df["month"] / 12)
check_df.drop(["period"], axis=1, inplace=True)

# Проверяем на наличие пропущенных значений
print(training_df.isnull().sum())
print(check_df.isnull().sum())

corr = training_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()

X = training_df.drop('living_persons', axis=1)
y = training_df['living_persons']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания на валидационной выборке
y_pred = model.predict(X_valid)

# Вычисление метрик
mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)

# Вывод метрик
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')

X_check = check_df.drop(["flat_id"], axis=1)

predictions = model.predict(X_check)
predictions = np.round(predictions).astype(int)
predictions[predictions < 0] = 0
result = pd.DataFrame({
    'flat_id': check_df['flat_id'],
    'living_persons': predictions
})

# Сохраняем с правильными разделителями
result.to_csv('result_base_rand_forest_regressor.csv', sep=';', index=False)