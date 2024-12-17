import streamlit as st
import pandas as pd
from groq import Groq
from langdetect import detect, DetectorFactory
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Seed agar hasil deteksi bahasa konsisten
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
    # Tambahkan instruksi eksplisit untuk bahasa
    if language == 'id':
        instruction = "Jawab dalam Bahasa Indonesia."
    elif language in ['zh-cn', 'zh-tw']:
        instruction = "回答时请使用中文。"
    else:
        instruction = "Respond in English."

    # Gabungkan instruksi dengan pesan user
    messages.insert(0, {"role": "system", "content": instruction})
    
    # Panggil Groq API
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

# Fungsi untuk analisis machine learning
def analyze_csv_with_ml(df, target_column, algorithm):
    X = df.drop(columns=[target_column])
    y = df[target_column]

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
    else:
        return "Algoritma tidak dikenali."

    # Latih model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluasi
    return {"MSE": mean_squared_error(y_test, predictions), "R2": r2_score(y_test, predictions)}

# Streamlit UI
st.set_page_config(page_title="Analisis Stunting", layout="wide")
st.title("📊 Analisis Stunting dengan AI dan Machine Learning")

# Opsi input: manual atau unggah CSV
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
        algorithm = st.selectbox("Pilih Algoritma ML:", ["Regresi Linear", "Random Forest"])
        target_column = st.selectbox("Pilih Target Kolom:", df.columns)

        if st.button("Analisis CSV"):
            with st.spinner("Menganalisis data..."):
                ml_result = analyze_csv_with_ml(df, target_column, algorithm)
                st.subheader("Hasil Machine Learning:")
                st.write(f"**Algoritma**: {algorithm}")
                st.write(f"**Mean Squared Error (MSE)**: {ml_result['MSE']}")
                st.write(f"**R2 Score**: {ml_result['R2']}")

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
