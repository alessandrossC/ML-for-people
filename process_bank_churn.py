import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет данные на обучающую и валидационную выборки.

    :param df: Исходный DataFrame
    :param target_col: Название целевого столбца
    :param test_size: Доля данных для валидации
    :return: Кортеж (train_x, val_x, train_y, val_y)
    """
    input_cols = [col for col in df.columns if col != target_col]

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=42
    )

    return (
        train_df[input_cols].copy(),
        val_df[input_cols].copy(),
        train_df[target_col].copy(),
        val_df[target_col].copy()
    )

def identify_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Определяет числовые и категориальные колонки.

    :param df: DataFrame
    :return: Кортеж (список числовых колонок, список категориальных колонок)
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def preprocess_train_data(
    train_x: pd.DataFrame,
    val_x: pd.DataFrame,
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Препроцессинг данных: обработка пропусков, кодирование категориальных признаков, масштабирование.

    :param train_x: Обучающие данные (признаки)
    :param val_x: Валидационные данные (признаки)
    :param scaler_numeric: Нужно ли масштабировать числовые признаки
    :return: Кортеж (обработанные train_x, val_x, pipeline)
    """
    numeric_cols, categorical_cols = identify_column_types(train_x)

    # Обработка пропущенных значений в числовых колонках
    imputer = SimpleImputer(strategy='mean')
    train_x[numeric_cols] = imputer.fit_transform(train_x[numeric_cols].copy())
    val_x[numeric_cols] = imputer.transform(val_x[numeric_cols].copy())

    # Заполняем NaN в категориальных колонках
    train_x[categorical_cols] = train_x[categorical_cols].copy().fillna("Unknown")
    val_x[categorical_cols] = val_x[categorical_cols].copy().fillna("Unknown")

    # Кодирование категориальных признаков
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_x[categorical_cols])  # Обучаем только на train!

    train_encoded = encoder.transform(train_x[categorical_cols])
    val_encoded = encoder.transform(val_x[categorical_cols])

    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    # Препроцессинг (масштабирование + OneHotEncoding)
    if scaler_numeric:
        scaler = MinMaxScaler()
        train_x[numeric_cols] = scaler.fit_transform(train_x[numeric_cols])
        val_x[numeric_cols] = scaler.transform(val_x[numeric_cols])

    # Создаем итоговый DataFrame
    train_x = np.hstack((train_x[numeric_cols].values, train_encoded))
    val_x = np.hstack((val_x[numeric_cols].values, val_encoded))

    all_columns = numeric_cols + encoded_cat_cols

    train_x = pd.DataFrame(train_x, columns=all_columns)
    val_x = pd.DataFrame(val_x, columns=all_columns)

    return train_x, val_x, encoder

def process_data(raw_df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Полный процесс предобработки данных: удаление лишних колонок, разбиение на train/val, препроцессинг.

    :param raw_df: Исходный DataFrame
    :param scaler_numeric: Нужно ли масштабировать числовые признаки
    :return: Словарь с train_x, train_y, val_x, val_y
    """
    # Удаляем ненужные колонки
    raw_df = raw_df.drop(columns=['id', 'CustomerId', 'Surname'], errors='ignore')

    # Разделяем данные
    train_x, val_x, train_y, val_y = split_data(raw_df, target_col='Exited')

    # Применяем предобработку
    train_x, val_x, encoder = preprocess_train_data(train_x, val_x, scaler_numeric)

    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'encoder': encoder
    }

def preprocess_new_data(new_data: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Обрабатывает новые данные с использованием обученного энкодера.

    :param new_data: Новый DataFrame с признаками
    :param encoder: Обученный OneHotEncoder
    :return: Обработанный DataFrame
    """
    numeric_cols, categorical_cols = identify_column_types(new_data)

    # Обработка пропущенных значений
    imputer = SimpleImputer(strategy='mean')
    new_data[numeric_cols] = imputer.fit_transform(new_data[numeric_cols].copy())

    # Заполняем NaN в категориальных колонках
    new_data[categorical_cols] = new_data[categorical_cols].copy().fillna("Unknown")

    # Применяем препроцессинг
    new_encoded = encoder.transform(new_data[categorical_cols])

    # Получаем обновленные названия колонок
    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    all_columns = numeric_cols + encoded_cat_cols

    # Преобразуем в DataFrame
    new_data = pd.DataFrame(np.hstack((new_data[numeric_cols].values, new_encoded)), columns=all_columns)

    return new_data
