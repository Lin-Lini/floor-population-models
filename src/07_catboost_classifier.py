import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

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
    """
    if df['period'].dtype != 'datetime64[ns]':
        raise ValueError("Столбец 'period' должен иметь тип datetime64[ns]. Проверьте преобразование даты.")

    df['month'] = df['period'].dt.month
    df['day_of_week'] = df['period'].dt.dayofweek
    df['season'] = df['month'].apply(get_season)
    df = df.drop('period', axis=1)
    return df

def handle_missing_values(df, columns, strategy='mean'):
    """
    Обрабатывает пропущенные значения в указанных столбцах.
    """
    df['sewage'] = df['sewage'].fillna(df['cold_water']+df['hot_water'])
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
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

def add_additional_features(training, check, norms):
    """
    Добавляет дополнительные признаки на основе норм потребления, не зависящие от living_persons.
    """
    training = create_norm_features(training, norms=norms)
    check = create_norm_features(check, norms=norms)

    # Кодирование признака 'season'
    seasons = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
    training['season'] = training['season'].map(seasons)
    check['season'] = check['season'].map(seasons)

    return training, check

def handle_outliers(df, features):
    """
    Обрабатывает выбросы в данных методом IQR, капируя значения.
    """
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Капирование выбросов
        df[feature] = np.where(df[feature] < lower_bound, lower_bound,
                               np.where(df[feature] > upper_bound, upper_bound, df[feature]))
    return df

def visualize_outliers_before_after(training, features, title_suffix='до обработки'):
    """
    Визуализирует выбросы до и после обработки.
    """
    for feature in features:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=training[feature])
        plt.title(f'Boxplot для {feature} {title_suffix}')
        plt.show()

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

def prepare_data(training, check, norms):
    """
    Подготавливает данные: добавляет признаки, обрабатывает пропущенные значения, обрабатывает выбросы, масштабирует.
    """
    # Извлечение признаков из даты
    training = extract_date_features(training)
    check = extract_date_features(check)

    # Обработка пропущенных значений
    columns_with_missing = ['sewage']
    training = handle_missing_values(training, columns_with_missing, strategy='mean')
    check = handle_missing_values(check, columns_with_missing, strategy='mean')

    # Добавление дополнительных признаков
    training, check = add_additional_features(training, check, norms)

    # Обработка выбросов
    numeric_features = ['cold_water', 'hot_water', 'power_supply', 'sewage']

    # Визуализация до обработки выбросов
    print("Визуализация выбросов до обработки:")
    visualize_outliers_before_after(training, numeric_features, title_suffix='до обработки')

    # Обработка выбросов
    training = handle_outliers(training, numeric_features)
    check = handle_outliers(check, numeric_features)

    # Визуализация после обработки выбросов
    print("Визуализация выбросов после обработки:")
    visualize_outliers_before_after(training, numeric_features, title_suffix='после обработки')

    # Масштабирование признаков
    X = training.drop(['flat_id', 'living_persons'], axis=1)
    y = training['living_persons']

    X_check = check.drop(['flat_id'], axis=1)

    X_scaled, X_check_scaled, scaler = scale_features(X, X_check)

    return X_scaled, y, X_check_scaled, scaler

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

def train_and_evaluate_catboost(X_train, y_train, X_valid, y_valid):
    """
    Обучает и оценивает CatBoost модель с фиксированными гиперпараметрами.
    Возвращает обученную модель.
    """
    print("\nОбучение CatBoost...")

    # Установка фиксированных гиперпараметров
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_state=42,
        verbose=0
    )

    # Обучение модели с ранней остановкой
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50)

    # Оценка модели
    evaluate_model(cat_model, X_valid, y_valid, title_prefix='CatBoost')

    return cat_model

def evaluate_model(model, X_valid, y_valid, title_prefix='Model'):
    """
    Оценивает модель на валидационной выборке и выводит метрики.
    """
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred, average='weighted')

    print(f"{title_prefix} - Accuracy: {accuracy:.4f}")
    print(f"{title_prefix} - F1 Score: {f1:.4f}")
    print(classification_report(y_valid, y_pred))

    # Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Истинные значения')
    plt.title(f'{title_prefix} - Матрица ошибок')
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

    # Убедимся, что значения целочисленные и в диапазоне от 1 до 7
    result['living_persons'] = result['living_persons'].astype(int)
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

def save_model_and_transformer(model, scaler, directory='saved_models'):
    """
    Сохраняет обученную модель и трансформер в указанную директорию.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Сохранение модели CatBoost
    model_path = os.path.join(directory, 'catboost_model.cbm')
    model.save_model(model_path)
    print(f"Модель сохранена по пути: {model_path}")

    # Сохранение трансформера PowerTransformer
    scaler_path = os.path.join(directory, 'power_transformer.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Трансформер сохранен по пути: {scaler_path}")

def load_model_and_transformer(model_path, scaler_path):
    """
    Загружает сохранённую модель и трансформер.
    """
    # Загрузка модели CatBoost
    model = CatBoostClassifier()
    model.load_model(model_path)
    print(f"Модель загружена из: {model_path}")

    # Загрузка трансформера PowerTransformer
    scaler = joblib.load(scaler_path)
    print(f"Трансформер загружен из: {scaler_path}")

    return model, scaler

if __name__ == "__main__":
    # Укажите пути к файлам данных
    training_path = 'training.csv'  # Замените на полный путь, если необходимо
    check_path = 'check.csv'        # Замените на полный путь, если необходимо

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
    X_scaled, y, X_check_scaled, scaler = prepare_data(training_df, check_df, norms)

    # Разделение данных и балансировка классов
    X_train_res, X_valid, y_train_res, y_valid = split_and_balance(X_scaled, y)

    # Обучение и оценка модели CatBoost
    cat_best = train_and_evaluate_catboost(X_train_res, y_train_res, X_valid, y_valid)

    # Анализ важности признаков
    print("\nАнализ важности признаков для CatBoost:")
    analyze_feature_importance(cat_best, X_scaled.columns, model_name='CatBoost')

    # Сохранение модели и трансформера
    print("\nСохранение модели и трансформера...")
    save_model_and_transformer(cat_best, scaler, directory='saved_models')

    # Предсказание и сохранение результатов
    print("\nПредсказание на контрольной выборке и сохранение результатов...")
    try:
        predict_and_save(cat_best, X_check_scaled, check_df, output_path='result_experiment_catboost_classification.csv')
    except ValueError as e:
        print("Произошла ошибка при сохранении предсказаний:", e)
    except Exception as e:
        print("Произошла неожиданная ошибка:", e)

    print("\nПроцесс завершён успешно!")