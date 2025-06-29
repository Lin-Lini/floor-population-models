import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

training = pd.read_csv('training.csv', sep=';', decimal=',')
check = pd.read_csv('check.csv', sep=';', decimal=',')

print(training.shape)
print(training.head())
print(training.dtypes)
print(training.isnull().sum())

training['sewage'].fillna(training['cold_water'] + training['hot_water'], inplace=True)
check['sewage'].fillna(check['cold_water'] + check['hot_water'], inplace=True)
print(training.isnull().sum())

training['period'] = pd.to_datetime(training['period'], format='%d.%m.%Y')
check['period'] = pd.to_datetime(check['period'], format='%d.%m.%Y')

for df in [training, check]:
    df['month'] = df['period'].dt.month
    df['year'] = df['period'].dt.year

training.drop('period', axis=1, inplace=True)
check.drop('period', axis=1, inplace=True)

sns.countplot(x='living_persons', data=training)
plt.title('Распределение количества проживающих')
plt.show()

X = training.drop(['flat_id', 'living_persons'], axis=1)
y = training['living_persons']

X_check = check.drop(['flat_id'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_check_scaled = scaler.transform(X_check)
joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

print(f'Лучшие параметры: {grid_search.best_params_}')
print(f'Лучшая точность: {grid_search.best_score_}')

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("Отчет классификации:")
print(classification_report(y_test, y_pred))

# Матрица ошибок
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанные значения')
plt.ylabel('Истинные значения')
plt.title('Матрица ошибок')
plt.show()

# Предсказание количества проживающих
check_predictions = best_rf.predict(X_check_scaled)

# Добавим предсказания в DataFrame
result = pd.DataFrame({
    'flat_id': check['flat_id'],
    'living_persons': check_predictions
})

# Убедимся, что значения целочисленные и в диапазоне от 1 до 7
result['living_persons'] = result['living_persons'].astype(int)
result['living_persons'] = result['living_persons'].clip(1, 7)

# Просмотр первых строк результата
print(result.head())

result.to_csv('result_experiment_random_forest_gridSearch.csv', sep=';', index=False)