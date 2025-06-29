import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# Функции для обработки данных и инженерии признаков

def get_season(month):
    """
    Возвращает сезон на основе месяца.
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def load_data(training_path, check_path):
    """
    Загружает обучающую и контрольную выборки из CSV файлов и преобразует столбец 'period' в datetime.
    """
    training = pd.read_csv(training_path, sep=';', decimal=',')
    check = pd.read_csv(check_path, sep=';', decimal=',')

    # Преобразование столбца 'period' в datetime для обучающей выборки
    training["period"] = pd.to_datetime(training["period"], format="%d.%m.%Y", errors='coerce')

    # Преобразование столбца 'period' в datetime для контрольной выборки
    check["period"] = pd.to_datetime(check["period"], format="%d.%m.%Y", errors='coerce')

    return training, check

def initial_analysis(training, check):
    """
    Выполняет первичный анализ данных: выводит форму, типы данных и количество пропусков.
    """
    print("Форма обучающей выборки:", training.shape)
    print("Форма контрольной выборки:", check.shape)
    print("\nПервичные данные обучающей выборки:")
    print(training.head())

    print("\nТипы данных обучающей выборки:")
    print(training.dtypes)
    print("\nПропущенные значения в обучающей выборке:")
    print(training.isnull().sum())

    print("\nПропущенные значения в контрольной выборке:")
    print(check.isnull().sum())

    # Анализ распределения целевой переменной
    sns.countplot(x='living_persons', data=training)
    plt.title('Распределение количества проживающих')
    plt.xlabel('Количество проживающих')
    plt.ylabel('Количество квартир/домов')
    plt.show()

def extract_date_features(df):
    """
    Извлекает дополнительные признаки из столбца даты и добавляет признак сезона.
    Удаляет ненужные признаки.
    """
    if df['period'].dtype != 'datetime64[ns]':
        raise ValueError("Столбец 'period' должен иметь тип datetime64[ns]. Проверьте преобразование даты.")

    df['month'] = df['period'].dt.month
    df['day_of_week'] = df['period'].dt.dayofweek
    df['season'] = df['month'].apply(get_season)
    # Удаляем столбец 'period'
    df = df.drop(['period'], axis=1)
    return df

def handle_missing_sewage(df):
    """
    Заполняет пропущенные значения в столбце 'sewage' суммой 'hot_water' и 'cold_water'.
    """
    df['sewage'] = df['sewage'].fillna(df['hot_water'] + df['cold_water'])
    return df

def create_norm_features(df, norms=None):
    """
    Создаёт признаки на основе норм потребления, не зависящие от living_persons.
    """
    if norms is None:
        norms = {
            'hot_water': 4.745,       # м³/чел./мес.
            'cold_water': 6.935,      # м³/чел./мес.
            'sewage': 11.68,          # м³/чел./мес.
            'power_supply': 120       # кВт·ч/чел./мес.
        }

    # Создание признаков отклонения от нормы на общий потребление
    for resource, norm in norms.items():
        df[f'deviation_{resource}'] = df[resource] - norm

    # Создание признаков отношения потребления к норме
    for resource, norm in norms.items():
        df[f'rate_{resource}'] = df[resource] / norm

    return df

def add_additional_features(df, norms):
    """
    Добавляет дополнительные признаки на основе норм потребления, не зависящие от living_persons.
    """
    df = create_norm_features(df, norms=norms)

    # Кодирование признака 'season'
    seasons = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
    df['season'] = df['season'].map(seasons)

    return df

def handle_outliers(df, features, multiplier=3.0):
    """
    Обрабатывает выбросы в данных методом IQR, заменяя значения за пределами границ на сами границы.
    Используется увеличенный множитель для менее агрессивного определения выбросов.

    Parameters:
    - df: DataFrame
    - features: список признаков для обработки выбросов
    - multiplier: множитель для определения границ (по умолчанию 3.0)

    Returns:
    - DataFrame с обработанными выбросами
    """
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        # Заменяем выбросы на пределы
        df[feature] = np.where(df[feature] < lower_bound, lower_bound,
                               np.where(df[feature] > upper_bound, upper_bound, df[feature]))
    return df

def visualize_outliers_before_after(df, features, title_suffix='до обработки'):
    """
    Визуализирует выбросы до и после обработки.
    """
    for feature in features:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[feature])
        plt.title(f'Boxplot для {feature} {title_suffix}')
        plt.show()

def add_aggregated_features(df):
    """
    Добавляет агрегированные признаки, такие как среднее потребление за последние 3 месяца.
    """
    df = df.sort_values(['flat_id', 'month'])
    df['avg_cold_water_last_3'] = df.groupby('flat_id')['cold_water'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['avg_hot_water_last_3'] = df.groupby('flat_id')['hot_water'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['avg_power_supply_last_3'] = df.groupby('flat_id')['power_supply'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['avg_sewage_last_3'] = df.groupby('flat_id')['sewage'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    return df

def create_lag_features(df, lag=1):
    """
    Создаёт лаговые признаки для указанных потреблений.
    """
    lag_features = ['hot_water', 'cold_water', 'sewage', 'power_supply']
    for feature in lag_features:
        df[f'{feature}_lag_{lag}'] = df.groupby('flat_id')[feature].shift(lag)
    return df

def create_rolling_features(df, window=3):
    """
    Создаёт скользящие средние для указанных потреблений.
    """
    rolling_features = ['hot_water', 'cold_water', 'sewage', 'power_supply']
    for feature in rolling_features:
        df[f'{feature}_rolling_mean_{window}'] = df.groupby('flat_id')[feature].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    return df

def create_interaction_features(df):
    """
    Создаёт взаимодействия между признаками.
    """
    # Пример взаимодействий
    df['hot_to_cold_water_ratio'] = df['hot_water'] / (df['cold_water'] + 1e-5)
    df['total_water'] = df['hot_water'] + df['cold_water']
    return df

def add_season_dummies(df):
    """
    Добавляет бинарные признаки для каждого сезона.
    """
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    for season in seasons:
        df[f'season_{season}'] = (df['season'] == season).astype(int)
    return df

def scale_features(X, X_check):
    """
    Масштабирует признаки с помощью PowerTransformer.
    """
    pt = PowerTransformer()
    X_transformed = pt.fit_transform(X)
    X_check_transformed = pt.transform(X_check)

    # Преобразуем обратно в DataFrame для удобства
    X_transformed = pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
    X_check_transformed = pd.DataFrame(X_check_transformed, columns=X.columns, index=X_check.index)

    return X_transformed, X_check_transformed, pt

def fill_missing_values(df):
    """
    Заполняет пропущенные значения средними значениями по признакам.
    """
    return df.fillna(df.mean())

def prepare_data(training, check, norms):
    """
    Подготавливает данные: объединяет обучающую и контрольную выборки для создания признаков,
    обрабатывает пропущенные значения, обрабатывает выбросы (только в обучающей выборке), масштабирует.
    Возвращает подготовленные обучающие и контрольные выборки.
    """
    # Добавляем флаг, чтобы позже разделить обратно
    training['is_train'] = 1
    check['is_train'] = 0
    check['living_persons'] = np.nan  # Добавляем целевую переменную для унификации

    # Объединяем обучающую и контрольную выборки
    combined = pd.concat([training, check], ignore_index=True)

    # Извлечение признаков из даты
    combined = extract_date_features(combined)

    # Заполнение пропущенных значений в 'sewage' суммой 'hot_water' и 'cold_water'
    combined = handle_missing_sewage(combined)

    # Добавление агрегированных признаков
    combined = add_aggregated_features(combined)

    # Добавление дополнительных признаков
    combined = add_additional_features(combined, norms)

    # Создание лаговых признаков
    combined = create_lag_features(combined, lag=1)

    # Создание скользящих средних
    combined = create_rolling_features(combined, window=3)

    # Создание взаимодействий между признаками
    combined = create_interaction_features(combined)

    # Добавление бинарных признаков для каждого сезона
    combined = add_season_dummies(combined)

    # Разделение обратно на обучающую и контрольную выборки
    training_prepared = combined[combined['is_train'] == 1].drop(['is_train'], axis=1)
    check_prepared = combined[combined['is_train'] == 0].drop(['is_train', 'living_persons'], axis=1)

    # Обработка выбросов только в обучающей выборке
    numeric_features = ['cold_water', 'hot_water', 'power_supply', 'sewage',
                        'avg_cold_water_last_3', 'avg_hot_water_last_3',
                        'avg_power_supply_last_3', 'avg_sewage_last_3',
                        'hot_water_lag_1', 'cold_water_lag_1', 'sewage_lag_1', 'power_supply_lag_1',
                        'hot_water_rolling_mean_3', 'cold_water_rolling_mean_3',
                        'sewage_rolling_mean_3', 'power_supply_rolling_mean_3',
                        'hot_to_cold_water_ratio', 'total_water']

    print("\nВизуализация выбросов до обработки (Обучающая выборка):")
    visualize_outliers_before_after(training_prepared, numeric_features, title_suffix='до обработки')

    # Обработка выбросов только в обучающей выборке
    training_prepared = handle_outliers(training_prepared, numeric_features, multiplier=3.0)

    print("Визуализация выбросов после обработки (Обучающая выборка):")
    visualize_outliers_before_after(training_prepared, numeric_features, title_suffix='после обработки')

    # Заполнение оставшихся пропущенных значений средними значениями
    training_prepared = fill_missing_values(training_prepared)
    check_prepared = fill_missing_values(check_prepared)

    # Удаление строк с NaN, если остались
    training_prepared = training_prepared.dropna()
    check_prepared = check_prepared.dropna()

    # Масштабирование признаков
    X_train_full = training_prepared.drop(['flat_id', 'living_persons'], axis=1)
    y_train_full = training_prepared['living_persons']

    X_check = check_prepared.drop(['flat_id'], axis=1)

    X_scaled, X_check_scaled, scaler = scale_features(X_train_full, X_check)

    # Проверка размеров после подготовки
    print(f"\nКоличество строк после подготовки данных:")
    print(f"Обучающая выборка: {X_scaled.shape[0]}")
    print(f"Контрольная выборка: {X_check_scaled.shape[0]}")

    return X_scaled, y_train_full, X_check_scaled, scaler


def split_and_balance(X, y):
    """
    Разделяет данные на обучающую и валидационную выборки и балансирует классы с помощью SMOTE.
    """
    # Разделение на обучающую и валидационную выборки
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Распределение классов в обучающей выборке:")
    print(y_train.value_counts())
    print("\nРаспределение классов в валидационной выборке:")
    print(y_valid.value_counts())

    # Балансировка классов с помощью SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\nРаспределение классов после балансировки:")
    print(pd.Series(y_train_res).value_counts())

    return X_train_res, X_valid, y_train_res, y_valid

from sklearn.model_selection import cross_val_score

def train_and_evaluate_catboost(X_train, y_train, X_valid, y_valid, best_params=None):
    """
    Обучает и оценивает CatBoost классификационную модель с кросс-валидацией.
    Возвращает обученную модель.
    """
    print("\nОбучение CatBoost Classifier...")

    if best_params is None:
        # Если лучшие параметры не предоставлены, используем дефолтные
        cat_model = CatBoostClassifier(  # Change to CatBoostClassifier
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_state=42,
            verbose=100
        )
    else:
        cat_model = CatBoostClassifier(  # Change to CatBoostClassifier
            iterations=best_params.get('iterations', 1000),
            depth=best_params.get('depth', 8),
            learning_rate=best_params.get('learning_rate', 0.05),
            l2_leaf_reg=best_params.get('l2_leaf_reg', 3),
            bagging_temperature=best_params.get('bagging_temperature', 4.65),
            random_state=42,
            verbose=100
        )

    # Кросс-валидация
    print("\nКросс-валидация для оценки модели...")
    scores = cross_val_score(cat_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Средняя точность кросс-валидации: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Обучение модели с ранней остановкой
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)

    # Оценка модели
    evaluate_classification_model(cat_model, X_valid, y_valid, title_prefix='CatBoost Classifier')

    return cat_model

def train_and_evaluate_random_forest(X_train, y_train, X_valid, y_valid):
    """
    Обучает и оценивает Random Forest классификационную модель с кросс-валидацией.
    Возвращает обученную модель.
    """
    print("\nОбучение Random Forest Classifier...")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Кросс-валидация
    print("\nКросс-валидация для оценки модели...")
    scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Средняя точность кросс-валидации: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_valid)

    # Оценка модели
    evaluate_classification_model(rf_model, X_valid, y_valid, title_prefix='Random Forest Classifier')

    return rf_model


def evaluate_classification_model(model, X_valid, y_valid, title_prefix='Model'):
    """
    Оценивает классификационную модель на валидационной выборке и выводит метрики.
    """
    y_pred = model.predict(X_valid)

    accuracy = (y_pred == y_valid).mean()
    print(f"{title_prefix} - Accuracy: {accuracy:.4f}")

    # Визуализация реальных vs предсказанных значений
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_valid, y=y_pred, alpha=0.5)
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    plt.title(f'{title_prefix} - Реальные vs Предсказанные значения')
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')  # Линия идеального соответствия
    plt.show()

def analyze_feature_importance(model, feature_names, model_name='Model'):
    """
    Анализирует и визуализирует важность признаков.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        print(f"Модель {model_name} не поддерживает метод feature_importances_.")
        return

    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_imp.head(20))
    plt.title(f'Важность признаков {model_name}')
    plt.show()

def predict_and_save(model, X_check, check_df, output_path='result.csv'):
    """
    Делает предсказания на контрольной выборке и сохраняет результаты в CSV файл.
    """
    check_predictions = model.predict(X_check)

    # Проверка формы предсказаний
    print("Форма предсказаний:", check_predictions.shape)

    # Преобразование предсказаний в 1D, если необходимо
    if check_predictions.ndim > 1:
        check_predictions = check_predictions.flatten()
        print("Предсказания преобразованы в 1D массив.")

    # Проверка наличия 'flat_id'
    if 'flat_id' not in check_df.columns:
        raise ValueError("Столбец 'flat_id' отсутствует в контрольной выборке.")

    # Извлечение 'flat_id'
    flat_id = check_df['flat_id']

    # Проверка формы 'flat_id'
    print("Форма 'flat_id':", flat_id.shape)

    # Преобразование 'flat_id' в 1D, если необходимо
    if flat_id.ndim > 1:
        flat_id = flat_id.values.flatten()
        print("'flat_id' преобразован в 1D массив.")

    # Создание DataFrame с результатами
    try:
        result = pd.DataFrame({
            'flat_id': flat_id,
            'living_persons': check_predictions
        })
    except ValueError as e:
        print("Ошибка при создании DataFrame:", e)
        print("Тип 'flat_id':", type(flat_id))
        print("Тип 'check_predictions':", type(check_predictions))
        raise

    # Округление предсказаний до ближайшего целого числа
    result['living_persons'] = result['living_persons'].round().astype(int)

    # Ограничение значений в диапазоне от 1 до 7
    result['living_persons'] = result['living_persons'].clip(1, 7)

    # Просмотр первых строк результата
    print(result.head())

    # Сохранение результатов
    try:
        result.to_csv(output_path, sep=';', index=False)
        print(f"Результаты сохранены в {output_path}")
    except Exception as e:
        print(f"Не удалось сохранить файл {output_path}: {e}")
        raise

def save_model_and_transformer(cat_model, rf_model, scaler, directory='saved_models'):
    """
    Сохраняет обученные модели и трансформер в указанную директорию.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Сохранение модели CatBoost
    cat_model_path = os.path.join(directory, 'catboost_model.cbm')
    cat_model.save_model(cat_model_path)
    print(f"Модель CatBoost сохранена по пути: {cat_model_path}")

    # Сохранение модели Random Forest
    rf_model_path = os.path.join(directory, 'random_forest_model.joblib')
    joblib.dump(rf_model, rf_model_path)
    print(f"Модель Random Forest сохранена по пути: {rf_model_path}")

    # Сохранение трансформера PowerTransformer
    scaler_path = os.path.join(directory, 'power_transformer.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Трансформер сохранен по пути: {scaler_path}")

def load_model_and_transformer(cat_model_path, rf_model_path, scaler_path):
    """
    Загружает сохранённые модели и трансформер.
    """
    # Определение расширения файла модели для определения типа модели
    _, ext_cat = os.path.splitext(cat_model_path)
    _, ext_rf = os.path.splitext(rf_model_path)

    if ext_cat == '.cbm':
        # Загрузка модели CatBoost
        cat_model = CatBoostClassifier()
        cat_model.load_model(cat_model_path)
        print(f"Модель CatBoost загружена из: {cat_model_path}")
    else:
        raise ValueError("Неподдерживаемый формат файла модели CatBoost.")

    if ext_rf == '.joblib':
        # Загрузка модели Random Forest
        rf_model = joblib.load(rf_model_path)
        print(f"Модель Random Forest загружена из: {rf_model_path}")
    else:
        raise ValueError("Неподдерживаемый формат файла модели Random Forest.")

    # Загрузка трансформера PowerTransformer
    scaler = joblib.load(scaler_path)
    print(f"Трансформер загружен из: {scaler_path}")

    return cat_model, rf_model, scaler

# Основной блок выполнения
if __name__ == "__main__":
    # Укажите пути к файлам данных
    training_path = 'training.csv'
    check_path = 'check.csv'

    # Загрузка данных
    training_df, check_df = load_data(training_path, check_path)

    # Первичный анализ
    initial_analysis(training_df, check_df)

    # Нормы потребления на одного человека
    norms = {
        'hot_water': 4.745,       # м³/чел./мес.
        'cold_water': 6.935,      # м³/чел./мес.
        'sewage': 11.68,          # м³/чел./мес.
        'power_supply': 120       # кВт·ч/чел./мес.
    }

    # Подготовка данных
    X_scaled, y_train_full, X_check_scaled, scaler = prepare_data(training_df, check_df, norms)

    print(f"Размер обучающей выборки после подготовки: {X_scaled.shape}")
    print(f"Размер контрольной выборки после подготовки: {X_check_scaled.shape}")

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_valid, y_train, y_valid = split_and_balance(X_scaled, y_train_full)

    # Обучение и оценка модели CatBoost Classifier
    cat_best = train_and_evaluate_catboost(X_train, y_train, X_valid, y_valid)

    # Обучение и оценка модели Random Forest Classifier
    rf_best = train_and_evaluate_random_forest(X_train, y_train, X_valid, y_valid)

    # Сохранение моделей и трансформера
    print("\nСохранение моделей и трансформера...")
    save_model_and_transformer(cat_best, rf_best, scaler, directory='saved_models')

    # Выбор лучшей модели на основе Accuracy
    cat_accuracy = (cat_best.predict(X_valid) == y_valid).mean()
    rf_accuracy = (rf_best.predict(X_valid) == y_valid).mean()

    if rf_accuracy > cat_accuracy:
        best_model = rf_best
        best_model_name = 'Random Forest Classifier'
    else:
        best_model = cat_best
        best_model_name = 'CatBoost Classifier'

    print(f"\nВыбор лучшей модели: {best_model_name} с Accuracy: {max(cat_accuracy, rf_accuracy):.4f}")

    # Предсказание и сохранение результатов
    print("\nПредсказание на контрольной выборке и сохранение результатов...")
    try:
        predict_and_save(best_model, X_check_scaled, check_df, output_path='result_experiment_catboost_or_randforest_classif.csv')
    except ValueError as e:
        print("Произошла ошибка при сохранении предсказаний:", e)
    except Exception as e:
        print("Произошла неожиданная ошибка:", e)

    print("\nПроцесс завершён успешно!")