import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from transformers import NAFillerTransformer, TorqueFeatureExtractor, MeasuredFeatureCleaner

st.title("Car price prediction")

def load_pickle_obj(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pickle_obj('model.pkl')
preprocessing_pipeline = load_pickle_obj('preprocessing_pipeline.pkl')

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
    year = st.number_input("Год выпуска", min_value=1900, format="%d")
    km_driven = st.number_input("Пробег, км", min_value=0, format="%d")
    fuel = st.selectbox("Тип топлива", ["", "Diesel", "Petrol", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Тип продавца", ["", "Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Трансмиссия", ["", "Manual", "Automatic"])
    owner = st.selectbox("Владелец", ["", "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
    mileage = st.text_input("Пробег (mileage), например '20 kmpl'")
    engine = st.text_input("Объем двигателя, например '1248 CC'")
    max_power = st.text_input("Максимальная мощность, например '74 bhp'")
    torque = st.text_input("Крутящий момент (torque), например '190Nm@ 2000rpm'")
    seats = st.number_input("Количество мест", min_value=1.0, step=1.0, format="%.0f")

    required_fields_filled = (
        name.strip() != "" and
        str(year).strip() != "" and year is not None and
        str(km_driven).strip() != "" and km_driven is not None and
        fuel.strip() != "" and
        seller_type.strip() != "" and
        transmission.strip() != "" and
        owner.strip() != "" and
        mileage.strip() != "" and
        engine.strip() != "" and
        max_power.strip() != "" and
        torque.strip() != "" and
        str(seats).strip() != "" and seats is not None
    )

    if not required_fields_filled:
        st.warning("Пожалуйста, заполните все обязательные параметры, чтобы предсказать цену")
    else:
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
            try:
                user_input_transformed = preprocessing_pipeline.transform(user_input)
                prediction = model.predict(user_input_transformed)
                st.write(f"Предсказанная цена: {prediction[0]:.2f}")
            except AttributeError as e:
                if "'str' object has no attribute 'transform'" in str(e):
                    st.error("Ошибка: объект preprocessing_pipeline оказался строкой. Проверьте файл preprocessing_pipeline.pkl — возможно он некорректно сохранён. Загрузите корректный сериализованный пайплайн.")
                else:
                    st.error(f"Ошибка при предобработке или предсказании: {e}")
            except Exception as e:
                st.error(f"Ошибка при предобработке или предсказании: {e}")

with st.expander("Загрузить данные из файла (файл из ДЗ: https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv)"):
    uploaded_file = st.file_uploader("Выберите файл", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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

        if st.button("Показать матрицу корреляций числовых признаков"):
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Корреляционная матрица")
            plt.tight_layout()
            st.pyplot(plt)

        try:
            st.write(f"Тип preprocessing_pipeline: {type(preprocessing_pipeline)}")
            if isinstance(preprocessing_pipeline, str):
                raise AttributeError("'str' object has no attribute 'transform'")
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
        except Exception as e:
            st.error(f"Ошибка при предобработке или предсказании для файла: {e}")
