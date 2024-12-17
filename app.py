import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Seed untuk konsistensi deteksi bahasa
DetectorFactory.seed = 0

# Inisialisasi Groq API
client = Groq(api_key="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")  # Ganti dengan API Key Anda

# Fungsi untuk deteksi bahasa
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Fungsi untuk meminta respons dari Groq API
def ask_groq(messages, language):
    if language == 'id':
        instruction = "Jawab dalam Bahasa Indonesia."
    elif language in ['zh-cn', 'zh-tw']:
        instruction = "回答时请使用中文。"
    else:
        instruction = "Respond in English."

    messages.insert(0, {"role": "system", "content": instruction})
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response

# Fungsi ringkasan CSV
def summarize_csv(df):
    return df.describe(include="all").transpose().fillna("N/A").to_string()

# Fungsi analisis data menggunakan ML
def analyze_csv_with_ml(df, target_column, algorithm):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Pisahkan kolom numerik dan non-numerik
    numeric_columns = X.select_dtypes(include=["number"]).columns
    non_numeric_columns = X.select_dtypes(exclude=["number"]).columns

    # Periksa dan hapus kolom kosong sepenuhnya (jika ada)
    X_numeric_valid = X[numeric_columns].dropna(axis=1, how='all')
    X_non_numeric_valid = X[non_numeric_columns].dropna(axis=1, how='all')

    # Imputasi kolom numerik dengan mean
    imputer_numeric = SimpleImputer(strategy="mean")
    X_numeric_imputed = imputer_numeric.fit_transform(X_numeric_valid)
    X_numeric = pd.DataFrame(X_numeric_imputed, columns=X_numeric_valid.columns, index=X.index)

    # Imputasi kolom non-numerik dengan modus
    imputer_non_numeric = SimpleImputer(strategy="most_frequent")
    X_non_numeric_imputed = imputer_non_numeric.fit_transform(X_non_numeric_valid)
    X_non_numeric = pd.DataFrame(X_non_numeric_imputed, columns=X_non_numeric_valid.columns, index=X.index)

    # Gabungkan kembali kolom numerik dan non-numerik
    X = pd.concat([X_numeric, X_non_numeric], axis=1)

    # Tambahkan kolom kosong yang dihapus sebelumnya
    for col in numeric_columns:
        if col not in X.columns:
            X[col] = pd.NA
    for col in non_numeric_columns:
        if col not in X.columns:
            X[col] = pd.NA

    # Konversi kategori ke numerik
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pilih model
    if algorithm == "Regresi Linear":
        model = LinearRegression()
    elif algorithm == "Random Forest":
        model = RandomForestRegressor()
    elif algorithm == "SVM":
        model = SVC(kernel='linear', max_iter=10000)
    elif algorithm == "Naive Bayes":
        model = GaussianNB()
    elif algorithm == "KNN":
        model = KNeighborsRegressor()
    else:
        return "Algoritma tidak dikenali."

    # Latih model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluasi
    return {
        "MSE": mean_squared_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
        "Actual": y_test,
        "Predicted": predictions,
    }

# Streamlit UI
st.set_page_config(page_title="Analisis Stunting", layout="wide")
st.title("📊 Analisis Stunting dengan AI dan Machine Learning")

# Opsi input
input_option = st.radio("Pilih jenis input:", ["Input Manual", "Unggah CSV"])

if input_option == "Input Manual":
    user_input = st.text_area("Masukkan Pertanyaan atau Data", "Apa penyebab stunting?")
    if st.button("Analisis Sekarang"):
        with st.spinner("Menganalisis..."):
            language = detect_language(user_input)
            st.info(f"Bahasa terdeteksi: **{language}**")
            messages = [{"role": "user", "content": user_input}]
            result = ask_groq(messages, language)
            st.subheader("Hasil Analisis:")
            st.write(result)

elif input_option == "Unggah CSV":
    uploaded_file = st.file_uploader("Unggah file CSV untuk dianalisis", type="csv")
    if uploaded_file:
        st.subheader("Pratinjau Data:")
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        # Input tambahan
        user_input = st.text_area("Masukkan Pertanyaan Tambahan", "Analisis data ini.")
        algorithm = st.selectbox("Pilih Algoritma ML:", ["Regresi Linear", "Random Forest", "SVM", "Naive Bayes", "KNN"])
        target_column = st.selectbox("Pilih Target Kolom:", df.columns)

        if st.button("Analisis CSV"):
            with st.spinner("Menganalisis data..."):
                ml_result = analyze_csv_with_ml(df, target_column, algorithm)
                st.subheader("Hasil Machine Learning:")
                st.write(f"**Algoritma**: {algorithm}")
                st.write(f"**Mean Squared Error (MSE)**: {ml_result['MSE']}")
                st.write(f"**R2 Score**: {ml_result['R2']}")

                # Grafik Hasil Prediksi dengan Plotly
                st.subheader("Grafik Perbandingan Prediksi dan Aktual")
                
                # Pastikan data dalam bentuk numpy array
                actual_values = (
                    ml_result['Actual'].to_numpy() if hasattr(ml_result['Actual'], 'to_numpy') else np.array(ml_result['Actual'])
                )
                predicted_values = (
                    ml_result['Predicted'] if isinstance(ml_result['Predicted'], np.ndarray) else np.array(ml_result['Predicted'])
                )

                # Buat grafik
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=actual_values,
                    mode='lines+markers',
                    name='Aktual',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=predicted_values,
                    mode='lines+markers',
                    name='Prediksi',
                    line=dict(color='red', dash='dot')
                ))
                fig.update_layout(
                    title="Perbandingan Nilai Aktual dan Prediksi",
                    xaxis_title="Indeks Data",
                    yaxis_title="Nilai Target",
                    legend_title="Legenda",
                    template="plotly_white"
                )
                st.plotly_chart(fig)

                # Menampilkan rumus matematika untuk algoritma yang dipilih
                if algorithm == "Regresi Linear":
                    st.subheader("Rumus Regresi Linear:")
                    st.markdown(r"""
                    $$y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$$
                    """)
                elif algorithm == "Random Forest":
                    st.subheader("Rumus Random Forest:")
                    st.markdown(r"""
                    $$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(x)$$
                    """)
                elif algorithm == "SVM":
                    st.subheader("Rumus SVM (Support Vector Machine):")
                    st.markdown(r"""
                    $$y = \text{sign}(\sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b)$$
                    """)
                elif algorithm == "Naive Bayes":
                    st.subheader("Rumus Naive Bayes:")
                    st.markdown(r"""
                    $$P(C|X) = \frac{P(C) \prod_{i=1}^{n} P(X_i|C)}{P(X)}$$
                    """)
                elif algorithm == "KNN":
                    st.subheader("Rumus K-Nearest Neighbors (KNN):")
                    st.markdown(r"""
                    $$y = \frac{1}{K} \sum_{i=1}^{K} y_i$$
                    """)
                
                # Ringkasan data & insight dengan Groq
                summary = summarize_csv(df)
                language = detect_language(user_input)
                messages = [
                    {"role": "user", "content": f"{user_input}\nRingkasan Data:\n{summary}"}
                ]
                result = ask_groq(messages, language)
                st.subheader("Insight dari AI:")
                st.write(result)

# Footer
st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit, Groq API, dan Machine Learning")
