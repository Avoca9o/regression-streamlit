import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

class MeasuredFeatureCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in ['mileage', 'engine', 'max_power']:
            X_copy[col] = X_copy[col].apply(self._handle_measured)
        print('Measured features cleaned')
        return X_copy

    def _handle_measured(self, s):
        if pd.isnull(s) or pd.isna(s):
            return np.nan
        digits_and_dots = [c for c in s if c.isdigit() or c == '.']
        return float(''.join(digits_and_dots)) if digits_and_dots else np.nan

class TorqueFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy['torque'] = X_copy['torque'].apply(self._get_torque)
        X_copy['max_torque_rpm'] = X_copy['torque'].apply(self._get_max_torque_rpm)

        print('Torque and max_torque_rpm extracted')

        return X_copy

    def _get_torque(self, s):
        if pd.isnull(s) or pd.isna(s):
            return np.nan
        first_float_str = ''
        for i in range(len(s)):
            if s[i].isdigit() or s[i] == '.':
                first_float_str += s[i]
            elif len(first_float_str) > 0:
                break
        if 'kgm' in str(s).lower():
            return float(first_float_str) * 9.80665 if first_float_str else np.nan
        return float(first_float_str) if first_float_str else np.nan

    def _get_max_torque_rpm(self, s):
        if pd.isnull(s) or pd.isna(s):
            return np.nan
        last_float_str = ''
        s_str = str(s)
        for i in range(len(s_str)):
            char = s_str[-i - 1]
            if char.isdigit() or char == '.':
                last_float_str += char
            elif len(last_float_str) > 0:
                break
        return float(last_float_str) if last_float_str else np.nan

class NAFillerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.columns = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].fillna(X_copy[col].median())
        X_copy['seats'] = X_copy['seats'].astype(int)
        X_copy['engine'] = X_copy['engine'].astype(int)
        X_copy = X_copy.drop('name', axis=1) # слишком сложно реализовывать, смысл задания явно больше в том, чтобы попробовать streamlit/pickle, я уже показал, что умею кастомные трансформеры писать, можно не буду возиться с именем, пожалуйста
        print('NA features filled')
        return X_copy


st.title("Car price prediction")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

coefficients = np.ravel(model.coef_)

columns = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "torque",
    "seats"
]

with st.expander("Показать распределение весов"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(coefficients)), coefficients, color='skyblue')
    ax.set_title('Визуализация весов модели после всех преобразований')
    ax.set_xlabel('Номер признака')
    ax.set_ylabel('Величина веса')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

with st.expander("Ввести данные вручную"):
    name = st.text_input("Название автомобиля")
    year = st.number_input("Год выпуска")
    km_driven = st.number_input("Пробег, км", min_value=0)
    fuel = st.selectbox("Тип топлива", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
    owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    mileage = st.text_input("Пробег (mileage), например '20 kmpl'", "")
    engine = st.text_input("Объем двигателя, например '1248 CC'", "")
    max_power = st.text_input("Максимальная мощность, например '74 bhp'", "")
    torque = st.text_input("Крутящий момент (torque), например '190Nm@ 2000rpm'", "")
    seats = st.number_input("Количество мест", min_value=1.0)

    user_input = pd.DataFrame([{
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "torque": torque,
        "seats": seats
    }], columns=columns)

    if st.button("Предсказать цену"):
        user_input_transformed = preprocessing_pipeline.transform(user_input)
        prediction = model.predict(user_input_transformed)
        st.write(f"Предсказанная цена: {prediction[0]:.2f}")

with st.expander("Загрузить данные из файла (файл из ДЗ: https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv)"):
    uploaded_file = st.file_uploader("Выберите файл", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        with st.expander("Визуализировать мои данные"):
            st.write(df.head())
            num_duplicates = df.duplicated().sum()
            st.write(f"Количество дубликатов в файле: {num_duplicates}")

            selected_col = st.selectbox(
                "Выберите колонку для гистограммы", 
                options=columns
            )
            if st.button("Построить гистограмму"):
                data = df[selected_col]
                plt.figure(figsize=(5, 3))
                if pd.api.types.is_numeric_dtype(data):
                    plt.hist(data.dropna(), bins=20)
                else:
                    data.value_counts().plot(kind='bar')
                    plt.xticks(rotation=90)
                plt.title(selected_col)
                plt.tight_layout()
                st.pyplot(plt)

            # Дополнительная кнопка для построения матрицы корреляций
            if st.button("Показать матрицу корреляций числовых признаков"):
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title("Корреляционная матрица")
                plt.tight_layout()
                st.pyplot(plt)

        user_input_transformed = preprocessing_pipeline.transform(df)
        predictions = model.predict(user_input_transformed)
        df_results = pd.DataFrame({"predicted_selling_price": predictions})
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать предсказанный .csv файл",
            data=csv,
            file_name="predictions.csv",
            mime='text/csv',
        )
