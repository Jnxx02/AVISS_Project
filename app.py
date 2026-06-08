import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================

st.set_page_config(
    page_title="A-VISS",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>

/* Sidebar Background */
[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 100%
    );
}

/* Sidebar Text */
[data-testid="stSidebar"] *{
    color:white;
}

/* Menu Radio */
div[role="radiogroup"] > label{
    background-color:#1e293b;
    padding:12px;
    border-radius:12px;
    margin-bottom:8px;
    transition:0.3s;
}

div[role="radiogroup"] > label:hover{
    background-color:#334155;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model/model_binary_subject_dependent.keras",
        compile=False,
        safe_mode=False
    )
    return model

# =====================================================
# LOAD DATA DEMO
# =====================================================

@st.cache_data
def load_demo_data():
    X = np.load("data/X_demo.npy")
    groups = np.load("data/groups_demo.npy")
    return X, groups

model = load_model()
X_data, groups = load_demo_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
<div style="padding-top:10px;">

<h1 style="
margin-bottom:0;
font-size:34px;
font-weight:700;
">
🩺 A-VISS
</h1>

<p style="
color:#94a3b8;
font-size:15px;
margin-top:0;
">
Stress Monitoring Prototype
</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Navigasi")

menu = st.sidebar.selectbox(
    "Pilih Halaman",
    [
        "🏠 Dashboard",
        "📈 Prediksi Stres",
        "🧠 Informasi Model",
        "📖 Tentang Penelitian"
    ],
    label_visibility="collapsed"
)

st.markdown("""
<style>

/* Selectbox */

[data-baseweb="select"] > div{
    background-color:#1e293b;
    border:none;
    border-radius:10px;
}

/* Hover */

[data-baseweb="select"] > div:hover{
    background-color:#334155;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**🧠 Model Aktif**

Binary Subject-Dependent CNN-LSTM

**📦 Versi**

v1.0
""")

from datetime import datetime

current_year = datetime.now().year

st.markdown("""
<style>

hr {
    margin-top: 10px !important;
    margin-bottom: 10px !important;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
    <div style="
        position:fixed;
        bottom:20px;
        left:20px;
        width:250px;
        font-size:12px;
        color:#94a3b8;
    ">
        © {current_year} Jonathan Kwan. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>

[data-testid="stSidebarContent"]{
    overflow-y: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "🏠 Dashboard":

    st.title("🩺 A-VISS")

    st.markdown("""
        ### Automated Vital Intelligence for Stress Surveillance

        Prototype klasifikasi stres berbasis wearable sensor menggunakan
        pendekatan **Subject-Dependent CNN-LSTM**.

        Dashboard ini digunakan untuk menampilkan hasil inferensi model
        CNN-LSTM terhadap data fisiologis yang diperoleh dari wearable sensor.
        """)

    st.markdown("---")

    st.subheader("📊 Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Awal", "11,5 Juta")

    with col2:
        st.metric("Subjek", "18")

    with col3:
        st.metric("Sensor", "6")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            "85%"
        )

    with col2:
        st.metric(
            "Weighted F1-Score",
            "86%"
        )

    with col3:
        st.metric(
            "Model",
            "CNN-LSTM"
        )

    st.markdown("---")

    st.markdown("### Pipeline Penelitian")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.info(
            """
            📊

            **Dataset**

            Nurse Stress
            Dataset
            """
        )

    with col2:
        st.info(
            """
            ⚙️

            **Preprocessing**

            Standardisasi
            Per Subjek
            """
        )

    with col3:
        st.info(
            """
            🪟

            **Windowing**

            60 Time Step
            Step = 5
            """
        )

    with col4:
        st.info(
            """
            🧠

            **CNN-LSTM**

            Deep Learning
            """
        )

    with col5:
        st.success(
            """
            🎯

            **Output**

            Stress
            Score
            """
        )

    st.info(
        """
        Dashboard ini menggunakan model Binary Subject-Dependent
        CNN-LSTM yang merupakan model terbaik pada penelitian A-VISS.
        """
    )

# =====================================================
# PREDIKSI STRES
# =====================================================

elif menu == "📈 Prediksi Stres":

    st.title("Prediksi Tingkat Stres")

    unique_ids = sorted(np.unique(groups))

    selected_id = st.selectbox(
        "Pilih ID Subjek",
        unique_ids
    )

    idx_id = np.where(groups == selected_id)[0]

    if len(idx_id) == 0:
        st.error(
            f"Tidak ditemukan data untuk ID {selected_id}"
        )
        st.stop()

    selected_window = st.slider(
        "Pilih Window Pengamatan",
        0,
        len(idx_id)-1,
        0
    )

    real_idx = idx_id[selected_window]

    sample = X_data[real_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "ID Perawat",
            str(selected_id)
        )

    with col2:
        st.metric(
            "Window Pengamatan",
            str(selected_window)
        )

    if st.button("Prediksi Stress"):

        input_data = np.expand_dims(sample, axis=0)

        prob = float(
            model.predict(
                input_data,
                verbose=0
            )[0][0]
        )

        score = round(prob * 100, 2)

        st.markdown("---")

        # =================================================
        # HASIL PREDIKSI (VERSI KPI CARD)
        # =================================================

        st.subheader("Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Stress Score",
                f"{score:.1f}%"
            )

        with col2:

            if score >= 70:
                st.error("🚨 HIGH STRESS")

            elif score >= 40:
                st.warning("⚠️ WASPADA")

            else:
                st.success("✅ NORMAL")

        # Progress Bar
        st.progress(score / 100)

        # Ringkasan Hasil
        if score >= 70:

            st.markdown(
                f"""
                ### 🚨 Tingkat Stres Tinggi

                Model CNN-LSTM menghasilkan
                **Stress Score sebesar {score:.1f}%**
                yang mengindikasikan kategori High Stress.
                """
            )

        elif score >= 40:

            st.markdown(
                f"""
                ### ⚠️ Tingkat Stres Sedang

                Model CNN-LSTM menghasilkan
                **Stress Score sebesar {score:.1f}%**
                yang mengindikasikan kategori Waspada.
                """
            )

        else:

            st.markdown(
                f"""
                ### ✅ Tingkat Stres Normal

                Model CNN-LSTM menghasilkan
                **Stress Score sebesar {score:.1f}%**
                yang mengindikasikan kategori Normal.
                """
            )

        st.markdown("""
        ### Kategori Stress Score

        | Rentang Skor | Kategori |
        |--------------|----------|
        | 0 – 40 | Normal |
        | 40 – 70 | Waspada |
        | 70 – 100 | High Stress |
        """)

        st.markdown("---")

        # =================================================
        # VISUALISASI SENSOR
        # =================================================

        df_sensor = pd.DataFrame(
            sample,
            columns=[
                "HR",
                "EDA",
                "TEMP",
                "X",
                "Y",
                "Z"
            ]
        )

        df_plot = df_sensor[
            ["HR", "EDA", "TEMP"]
        ].copy()

        df_plot.columns = [
            "HR (Z-Score)",
            "EDA (Z-Score)",
            "TEMP (Z-Score)"
        ]

        fig = px.line(
            df_plot,
            title="Sinyal Fisiologis Terstandarisasi (Z-Score)"
        )

        fig.update_layout(
            xaxis_title="Time Step",
            yaxis_title="Z-Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            """
            Nilai sensor yang ditampilkan merupakan hasil
            standardisasi per subjek menggunakan StandardScaler.
            
            Nilai negatif tidak menunjukkan sensor bernilai negatif,
            melainkan menunjukkan bahwa nilai tersebut berada di bawah
            rata-rata baseline fisiologis subjek.
            """
        )

# =====================================================
# INFORMASI MODEL
# =====================================================

elif menu == "🧠 Informasi Model":

    st.title("🧠 Informasi Model")

    st.markdown("""
    ### Model Terbaik Penelitian

    Binary Subject-Dependent CNN-LSTM
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            "85%"
        )

    with col2:
        st.metric(
            "Weighted F1",
            "86%"
        )

    with col3:
        st.metric(
            "Kelas",
            "Binary"
        )

    st.markdown("---")

    info = pd.DataFrame({

        "Parameter": [

            "Arsitektur",

            "Skenario",

            "Input Shape",

            "Window Size",

            "Step Size",

            "Jumlah Sensor",

            "Jumlah Kelas",

            "Output Layer",

            "Loss Function"

        ],

        "Nilai": [

            "CNN-LSTM",

            "Subject-Dependent",

            "(60,6)",

            "60",

            "5",

            "6",

            "Binary",

            "Sigmoid",

            "Binary Crossentropy"

        ]

    })

    st.table(info)

    st.markdown("---")

    st.subheader("Struktur Model")

    st.code("""
Input (60,6)
│
├── Conv1D (32)
├── Batch Normalization
├── MaxPooling1D
├── Dropout
│
├── Conv1D (64)
├── Batch Normalization
├── MaxPooling1D
│
├── LSTM (32)
├── Dropout
│
├── Dense (16)
└── Dense (1, Sigmoid)
""")

# =====================================================
# TENTANG PENELITIAN
# =====================================================

elif menu == "📖 Tentang Penelitian":

    st.title("📖 Tentang Penelitian")

    st.markdown("""
    ## 🩺 A-VISS v1.0

    **Automated Vital Intelligence for Stress Surveillance**

    Sistem prototipe deteksi stres perawat berbasis wearable sensor
    menggunakan pendekatan Deep Learning CNN-LSTM.
    """)

    st.markdown("---")

    # =====================================================
    # JUDUL PENELITIAN
    # =====================================================

    st.subheader("🎓 Judul Penelitian")

    st.info("""
    **RANCANG BANGUN PROTOTYPE A-VISS UNTUK KLASIFIKASI STRES PERAWAT
    MENGGUNAKAN PENDEKATAN SUBJECT-DEPENDENT CNN-LSTM**
    """)

    st.markdown("---")

    # =====================================================
    # RINGKASAN PENELITIAN
    # =====================================================

    st.subheader("📊 Ringkasan Penelitian")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Awal",
            "11,5 Juta"
        )

    with col2:
        st.metric(
            "Subjek",
            "18 Perawat"
        )

    with col3:
        st.metric(
            "Sensor",
            "6 Fitur"
        )

    with col4:
        st.metric(
            "Model",
            "CNN-LSTM"
        )

    st.markdown("---")

    # =====================================================
    # FITUR SENSOR
    # =====================================================

    st.subheader("⌚ Sensor Wearable yang Digunakan")

    sensor1, sensor2, sensor3 = st.columns(3)

    with sensor1:
        st.success("""
        ❤️ Heart Rate (HR)

        Mengukur detak jantung.
        """)

        st.success("""
        💧 EDA

        Electrodermal Activity
        """)

    with sensor2:
        st.success("""
        🌡️ TEMP

        Skin Temperature
        """)

        st.success("""
        📍 X

        Accelerometer Axis-X
        """)

    with sensor3:
        st.success("""
        📍 Y

        Accelerometer Axis-Y
        """)

        st.success("""
        📍 Z

        Accelerometer Axis-Z
        """)

    st.markdown("---")

    # =====================================================
    # PIPELINE
    # =====================================================

    st.subheader("⚙️ Pipeline Penelitian")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.info("📊\n\nDataset")

    with col2:
        st.info("⚙️\n\nStandardisasi\nPer Subjek")

    with col3:
        st.info("🪟\n\nSliding Window\n(60,5)")

    with col4:
        st.info("🧠\n\nCNN-LSTM")

    with col5:
        st.success("🎯\n\nStress Score")

    st.markdown("---")

    # =====================================================
    # HASIL TERBAIK
    # =====================================================

    st.subheader("🏆 Hasil Model Terbaik")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            "85%"
        )

    with col2:
        st.metric(
            "Weighted F1",
            "86%"
        )

    with col3:
        st.metric(
            "Skenario Terbaik",
            "Binary SD"
        )

    st.success("""
    Model terbaik yang diperoleh pada penelitian ini adalah
    **Binary Subject-Dependent CNN-LSTM**
    dengan Accuracy sebesar **85%**
    dan Weighted F1-Score sebesar **86%**.
    """)

    st.markdown("---")

    st.caption(
    "Prototype A-VISS for Stress Classification Using Subject-Dependent CNN-LSTM"
    )