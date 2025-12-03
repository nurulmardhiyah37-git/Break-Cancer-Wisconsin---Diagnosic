import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="wide",
    page_icon="🩺"
)
# LOAD MODEL & FEATURE LABELS
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# feature_names & feature_list dipakai untuk mengisi dummy features
try:
    with open("feature_names.pkl", "rb") as f:
        feature_labels = pickle.load(f)
except Exception:
    feature_labels = {}

with open("feature_list.pkl", "rb") as f:
    feature_list = pickle.load(f)


def get_malignant_probability(model, X):
    """
    Return probability (0-1) that sample is Malignant.
    Tries to locate 'malignant'/'M'/1 label in model.classes_.
    If predict_proba not available, return None.
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        mal_idx = None
        for i, c in enumerate(classes):
            s = str(c).lower()
            if s.startswith("m") or s.startswith("mal") or s == "1" or s == "1.0":
                mal_idx = i
                break
        if mal_idx is None and len(classes) == 2:
            mal_idx = 1
        if mal_idx is None:
            return None
        return float(probs[mal_idx])
    except Exception:
        return None

# CHATBOT TENAGA MEDIS (RULE-BASED)
def chatbot_medical(query):
    """Rule-based chatbot untuk tenaga medis — versi lengkap"""
    query = query.lower().strip()

    df = st.session_state.get("medical_df", None)

    if df is None:
        needs_data = ["ganas", "jinak", "pasien", "berapa", "jumlah", "data", "prediksi", "summary", "ringkas"]
        if any(k in query for k in needs_data):
            return "Silakan unggah file CSV terlebih dahulu agar saya dapat melakukan analisis data pasien Anda."
        else:
            return "Saya siap membantu setelah Anda mengunggah data pasien (CSV)."

    try:
        if "Prediction" not in df.columns:
            if all(feat in df.columns for feat in feature_list):
                df_input = df[feature_list]
                df["Prediction"] = model.predict(df_input)
            else:
                return "Kolom fitur lengkap tidak ditemukan di dataset. Pastikan CSV berisi semua fitur yang dibutuhkan."
    except Exception as e:
        return f"Gagal memproses prediksi otomatis: {e}"

    # 3. Statistik dasar
    total = len(df)
    if total == 0:
        return "Dataset kosong — tidak ada baris pasien yang bisa dianalisis."

    # Toleransi bila label bukan "M"/"B", normalisasikan kalau perlu
    pred_values = df["Prediction"].astype(str)
    ganas = (pred_values == "M").sum()
    jinak = (pred_values == "B").sum()
    # jika model menyimpan label lain (mis. 1/0), coba mapping
    if (ganas + jinak) == 0:
        if set(pred_values.unique()).issubset({"1", "0", "1.0", "0.0"}):
            ganas = (pred_values == "1").sum()
            jinak = (pred_values == "0").sum()

    try:
        persen_ganas = round((ganas / total) * 100, 2)
        persen_jinak = round((jinak / total) * 100, 2)
    except ZeroDivisionError:
        persen_ganas = persen_jinak = 0.0

    # A. PERTANYAAN TENTANG DATA PASIEN
    if ("berapa" in query and "total" in query and "pasien" in query) or ("total pasien" in query):
        return f"Total pasien dalam dataset Anda adalah **{total} pasien**."

    if "jumlah baris" in query or "jumlah data" in query:
        return f"Dataset mengandung **{total} baris data pasien**."

    if "jumlah kolom" in query or "berapa fitur" in query:
        return f"Dataset Anda memiliki **{df.shape[1]} kolom**, termasuk kolom prediksi jika tersedia."

    if "fitur apa saja" in query or "kolom apa saja" in query:
        cols = "\n".join([f"- {col}" for col in df.columns])
        return "Berikut daftar kolom pada dataset Anda:\n\n" + cols

    if "missing" in query or "kosong" in query or "null" in query:
        missing_series = df.isnull().sum()
        total_missing = int(missing_series.sum())
        if total_missing == 0:
            return "Tidak ada missing value pada dataset. Semua data tampak lengkap."
        else:
            top_missing = missing_series[missing_series > 0].sort_values(ascending=False).head(10)
            details = "\n".join([f"- {idx}: {val} nilai kosong" for idx, val in top_missing.items()])
            return f"Terdapat **{total_missing}** nilai kosong. Kolom dengan missing value terbanyak:\n{details}"

    if ("summary" in query) or ("ringkas" in query) or ("rangkuman" in query) or ("jelaskan data" in query) or ("jelaskan data saya" in query):
        return (
            f"Rangkuman singkat dataset Anda:\n\n"
            f"- Total pasien: **{total}**\n"
            f"- Tumor ganas (Malignant): **{ganas} pasien ({persen_ganas}%)**\n"
            f"- Tumor jinak (Benign): **{jinak} pasien ({persen_jinak}%)**\n"
        )

    # B. PERTANYAAN TENTANG ANALISIS PREDIKSI
    if "berapa" in query and "ganas" in query:
        return f"Terdapat **{ganas} pasien** dengan tumor ganas (Malignant)."

    if "berapa" in query and "jinak" in query:
        return f"Terdapat **{jinak} pasien** dengan tumor jinak (Benign)."

    if "persen" in query and "ganas" in query:
        return f"Persentase tumor ganas adalah **{persen_ganas}%** dari total pasien."

    if "persen" in query and "jinak" in query:
        return f"Persentase tumor jinak adalah **{persen_jinak}%** dari total pasien."

    if ("distribusi" in query or "bagaimana distribusi" in query or "plot distribusi" in query):
        # arahkan user ke UI karena chatbot rule-based simpel tidak menggambar grafik
        return "Saya bisa menampilkan distribusi prediksi di panel jika Anda melihat bagian 'Distribusi Prediksi' setelah upload file."

    if "model apa" in query or "algoritma apa" in query or "pakai model" in query:
        return "Model yang digunakan adalah **Random Forest Classifier** — dipilih karena kestabilan dan performanya pada dataset fitur numerik seperti WBCD."

    if ("kenapa" in query or "mengapa" in query) and ("prediksi" in query or "hasil" in query):
        return (
            "Prediksi dihasilkan berdasarkan pola fitur FNA (mis. radius, texture, perimeter, area). "
            "Nilai fitur yang jauh dari rentang normal cenderung mengarah ke klasifikasi ganas."
        )

    # contoh sederhana pola (heuristik)
    if "pola" in query or "insight" in query or "fitur berpengaruh" in query:
        top_feats = ["radius_worst", "area_mean", "perimeter_worst", "texture_mean"]
        present = [f for f in top_feats if f in df.columns]
        if present:
            return (
                "Berdasarkan pengetahuan umum WBCD, fitur yang sering berpengaruh adalah: "
                + ", ".join(present) +
                ".\nJika mau, saya bisa bantu analisis lebih lanjut (mis. perbandingan rata-rata antara kelas)."
            )
        else:
            return "Beberapa fitur penting tidak ditemukan di dataset untuk memberi insight fitur paling berpengaruh."

    # C. PERTANYAAN TENTANG VALIDITAS & EVALUASI MODEL
    if "valid" in query or "validitas" in query or ("akurasi" in query and "model" in query) or "akurasi" in query:
        return (
            "Model Random Forest yang digunakan umumnya memberikan akurasi tinggi pada dataset WBCD (sering di kisaran >90%). "
            "Namun, validitas di konteks Anda bergantung pada kualitas data (missing, distribusi fitur) dan apakah data berasal dari populasi yang sama."
        )

    if "bisa dipercaya" in query or "seberapa yakin" in query or "confidence" in query:
        return (
            "Model cukup dapat dipercaya sebagai alat screening awal. "
            "Untuk kepercayaan klinis, hasil prediksi harus dikonfirmasi oleh pemeriksaan lanjutan (biopsi/mammogram) dan pendapat ahli."
        )

    if "overfitting" in query or "over fit" in query:
        return (
            "Overfitting diatasi dengan pembagian train-test dan parameter tuning (mis. jumlah pohon). "
            "Untuk verifikasi, Anda bisa menjalankan cross-validation dan melihat perbedaan skor train vs test."
        )

    if "dataset apa" in query or "sumber data" in query:
        return "Dataset dasar yang terkait adalah **Wisconsin Breast Cancer Diagnostic (WBCD)** — dataset umum untuk tugas ini."

    # D. RULE TEKNIS (DEFAULT)
    tech_responses = {
        "precision": "Precision menunjukkan ketepatan model dalam memprediksi tumor ganas — dari semua yang diprediksi ganas, berapa yang benar-benar ganas.",
        "recall": "Recall menunjukkan kemampuan model menemukan seluruh kasus ganas (sensitivity).",
        "f1" : "F1 Score adalah rata-rata harmonik antara precision dan recall, cocok untuk dataset imbalance.",
        "random forest": "Random Forest bekerja dengan membuat banyak decision tree dan mengambil hasil voting terbanyak untuk prediksi akhir.",
        "csv": "Pastikan file CSV memiliki fitur numerik sesuai urutan feature_list yang digunakan saat pelatihan.",
        "error": "Kesalahan umum saat upload biasanya format kolom tidak sesuai atau terdapat nilai non-numeric pada kolom fitur."
    }

    for key, val in tech_responses.items():
        if key in query:
            return val

    # E. JAWABAN DEFAULT
    return "Maaf, saya belum memahami pertanyaan tersebut. Coba tanyakan tentang jumlah pasien, persentase ganas/jinak, missing value, atau validitas model."

# CHATBOT PASIEN 
def chatbot_patient(user_input, prediksi_user=None):
    user_input = user_input.lower()

    # RULE: CEMAS / TAKUT
    if any(k in user_input for k in ["takut", "cemas", "anxious", "khawatir"]):
        return (
            "Saya memahami rasa cemas itu wajar. Namun hasil model hanya sebagai alat bantu, "
            "yang menentukan tetap pemeriksaan dokter. Kamu tidak sendirian, saya di sini untuk membantu 😊"
        )
    # RULE: INFORMASI JENIS KANKER 
    elif any(k in user_input for k in ["apa itu kanker", "berbahaya", "ganas", "malignan", "benign"]):
        return (
            "Tumor jinak **benign** biasanya tidak menyebar, sedangkan tumor ganas **malignant** "
            "dapat berkembang dan menyerang jaringan lain. Pemeriksaan lanjutan seperti USG, mammogram, "
            "atau biopsi diperlukan untuk kepastian."
        )
    # RULE: GEJALA 
    elif any(k in user_input for k in ["gejala", "symptom", "tanda"]):
        return (
            "Beberapa gejala yang perlu diperhatikan:\n"
            "- Benjolan keras yang tidak bergerak\n"
            "- Perubahan bentuk payudara\n"
            "- Nyeri tidak biasa\n"
            "- Keluar cairan dari puting\n\n"
            "Namun diagnosis pasti harus melalui pemeriksaan dokter ya."
        )
    # RULE: SARAN SETELAH PREDIKSI
    elif "apa yang harus saya lakukan" in user_input:
        if prediksi_user == "Malignant":
            return (
                "Berdasarkan hasil prediksi model yang menunjukkan potensi keganasan, "
                "kamu sangat disarankan untuk melakukan pemeriksaan lanjutan seperti USG atau mammogram "
                "dan konsultasi dengan **dokter spesialis bedah onkologi**."
            )
        else:
            return (
                "Hasil prediksi cenderung jinak, tetapi tetap lakukan pemeriksaan rutin "
                "dan konsultasi jika muncul perubahan pada payudara."
            )
    # RULE: INFORMASI DOKTER SPESIALIS
    elif any(keyword in user_input for keyword in [
        "dokter spesialis", "spesialis kanker", "dokter onkologi",
        "onkolog", "dokter kanker", "spesialis payudara"
    ]):
        return (
            "Untuk pemeriksaan terkait kanker payudara, dokter yang tepat adalah:\n\n"
            "👩‍⚕️ **Dokter Spesialis Bedah Onkologi** – menangani operasi dan evaluasi tumor.\n"
            "🩺 **Dokter Spesialis Onkologi Medik** – menangani kemoterapi & terapi obat.\n"
            "🔬 **Dokter Spesialis Radiologi** – membaca hasil mammogram & USG.\n\n"
            "Jika hasil pemeriksaan menunjukkan potensi keganasan, sangat disarankan "
            "berkonsultasi dengan dokter bedah onkologi untuk evaluasi lanjutan."
        )
    #  RULE: DEFAULT 
    else:
        return (
            "Maaf, saya belum mengerti maksud pertanyaan Anda. "
            "Coba tanyakan tentang gejala, kecemasan, dokter spesialis, atau penjelasan prediksi ya 😊"
        )

# UI CHAT COMPONENT
def chat_ui(chat_history_key, chatbot_function, title):
    st.subheader(title)

    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    # show history
    for sender, msg in st.session_state[chat_history_key]:
        if sender == "user":
            st.markdown(f"🧑 **Anda:** {msg}")
        else:
            st.markdown(f"🤖 **Chatbot:** {msg}")

    user_input = st.text_input("Ketik pertanyaan Anda di sini:", key=f"input_{chat_history_key}")

    if st.button("Kirim", key=f"send_{chat_history_key}"):
        if user_input:
            st.session_state[chat_history_key].append(("user", user_input))
            reply = chatbot_function(user_input)
            st.session_state[chat_history_key].append(("bot", reply))
            st.rerun()

# APLIKASI UTAMA
st.title("🩺 Sistem Prediksi Kanker Payudara (Breast Cancer Classifier)")
st.write(
    """
    Selamat Datang di **Breast Cancer Prediction System**

    Silakan pilih **mode penggunaan** sesuai kebutuhan Anda:

    - mode **Tenaga Medis** — untuk menganalisis dataset, melihat ringkasan pasien, dan melakukan eksplorasi data.
    - mode **Pasien** — untuk melakukan *screening* awal sebelum pemeriksaan FNA.

    Pilih mode dibawah ini untuk melanjutkan.
    """
)

mode = st.radio("Pilih Mode:", ["Mode Tenaga Medis", "Mode Pasien"], horizontal=True)

# MODE TENAGA MEDIS
if mode == "Mode Tenaga Medis":
    st.header("📄 Silahkan Upload File Pemeriksaan (CSV)")

    file = st.file_uploader("Unggah file CSV hasil pemeriksaan FNA", type=["csv"])

    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            df = None

        if df is not None:
            st.session_state["medical_df"] = df  #SAVE DATA UNTUK CHATBOT

            st.success("File berhasil dibaca!")
            st.dataframe(df.head())

            # pastikan semua fitur ada sebelum prediksi
            if all(feat in df.columns for feat in feature_list):
                try:
                    df_input = df[feature_list]
                    pred = model.predict(df_input)
                    df["Prediction"] = pred
                    df["Pred_Description"] = df["Prediction"].replace({
                        "M": "Malignant (Ganas)",
                        "B": "Benign (Jinak)",
                        1: "Malignant (Ganas)",
                        0: "Benign (Jinak)"
                    })
                except Exception as e:
                    st.error(f"Gagal melakukan prediksi: {e}")
            else:
                st.warning("Beberapa fitur yang dibutuhkan model tidak ditemukan di CSV. Pastikan kolom sesuai feature_list.")
                st.session_state["medical_df"] = df

            st.subheader("📌 Hasil Prediksi")
            st.dataframe(st.session_state["medical_df"])

            # Simpan hasil untuk chatbot
            st.session_state["medical_df"] = st.session_state["medical_df"]

            # Download hasil prediksi 
            try:
                csv = st.session_state["medical_df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Hasil Prediksi",
                    data=csv,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Gagal membuat tombol download: {e}")

            st.subheader("📊 Distribusi Prediksi")
            # jika kolom Pred_Description ada, pakai itu, kalau tidak pakai Prediction
            if "Pred_Description" in st.session_state["medical_df"].columns:
                st.bar_chart(st.session_state["medical_df"]["Pred_Description"].value_counts())
            else:
                st.bar_chart(st.session_state["medical_df"]["Prediction"].value_counts())

    chat_ui("chat_medical", chatbot_medical, "💬 Chatbot Teknis (Untuk Tenaga Medis)")

# MODE PASIEN (SCREENING AWAL - BELUM FNA)
else:
    st.header("Mode Pasien")
    st.write("""
    Jika Anda merasakan benjolan tapi belum menjalani FNA/biopsi, isi form singkat berikut.
    model kami akan memberikan **indikasi awal** (bukan diagnosis) apakah benjolan cenderung jinak atau ganas.
    """)

    st.markdown("### Silahkan isi fitur sesuai dengan keadaan anda")

    col1, col2 = st.columns(2)

    with col1:
        size_cm = st.number_input("Ukuran benjolan (cm)", min_value=0.1, max_value=30.0, value=1.5, step=0.1,
                                  help="Perkiraan diameter benjolan berdasarkan yang Anda rasakan atau hasil pemeriksaan fisik.")
        edge = st.selectbox("Bentuk tepi benjolan", ["Halus (bulat/teratur)", "Tidak rata/bergelombang"],
                            help="Jika tepi tidak rata atau bergerigi, pilih opsi kedua.")
        hardness = st.selectbox("Kekerasan benjolan", ["Lunak", "Keras"],
                                help="Benjolan keras lebih mengkhawatirkan daripada benjolan lunak.")
        pain = st.radio("Apakah terasa nyeri?", ["Tidak", "Ya"])
        age = st.number_input("Usia Anda (tahun)", min_value=10, max_value=120, value=40, step=1)

    with col2:
        skin_change = st.radio("Adakah perubahan kulit di area benjolan?", ["Tidak", "Ya"],
                               help="Misal kulit cekung, kemerahan, atau menebal.")
        nipple_discharge = st.radio("Ada cairan keluar?", ["Tidak", "Ya"])
        axillary_swelling = st.radio("Ada pembengkakan kelenjar (ketiak)?", ["Tidak", "Ya"])
        family_history = st.radio("Riwayat keluarga kanker payudara?", ["Tidak", "Ya"])

    st.markdown("---")
    st.info(
    """
    **Catatan Penting:**  
    Hasil prediksi ini **bersifat indikatif** dan sebagai **indikasi awal**.  
    Untuk kepastian kondisi, silakan lakukan pemeriksaan **FNA/Biopsi**  
    serta segera konsultasikan dengan **dokter atau tenaga kesehatan profesional**.
    """
)

    if st.button("🔍 Dapatkan Indikasi Sekarang"):
        # Mapping heuristik: dari input pasien -> fitur mirip FNA
        dummy = {feat: 0.0 for feat in feature_list}
        radius = float(size_cm)
        dummy["radius_mean"] = 10.0 + radius * 2.5  

        # perimeter_mean proportional terhadap radius
        dummy["perimeter_mean"] = dummy["radius_mean"] * 6.0

        # area_mean proportional terhadap ukuran (perkiraan)
        dummy["area_mean"] = max(20.0, radius * 80.0)

        # texture_mean sedikit lebih tinggi jika tepi tidak rata
        dummy["texture_mean"] = 10.0 + (0.0 if edge.startswith("Halus") else 6.0)

        # smoothness_mean sedikit naik jika tepi tidak rata atau usia lebih tua
        dummy["smoothness_mean"] = 0.08 + (0.02 if edge.startswith("Tidak") else 0.0) + (0.005 if age > 50 else 0.0)

        # concavity/concave points increase
        if nipple_discharge == "Ya":
            dummy["concave points_mean"] = 0.05 + 0.03
            dummy["concavity_mean"] = 0.05 + 0.04
        else:
            dummy["concave points_mean"] = 0.02
            dummy["concavity_mean"] = 0.02

        # increase some 'worst' metrics if hardness == keras or skin changes or axillary swelling
        hardness_flag = 1 if hardness == "Keras" else 0
        risk_flag = hardness_flag + (1 if skin_change == "Ya" else 0) + (1 if axillary_swelling == "Ya" else 0) + (1 if family_history == "Ya" else 0)
        # map to some worst features if they exist
        if "radius_worst" in dummy:
            dummy["radius_worst"] = dummy["radius_mean"] + risk_flag * 5.0
        if "perimeter_worst" in dummy:
            dummy["perimeter_worst"] = dummy["perimeter_mean"] + risk_flag * 12.0
        if "area_worst" in dummy:
            dummy["area_worst"] = dummy["area_mean"] + risk_flag * 200.0
        if "texture_worst" in dummy:
            dummy["texture_worst"] = dummy.get("texture_worst", 12.0 + risk_flag * 2.0)

        # small adjustments from pain (pain tends to be non-specific; give small weight)
        if pain == "Ya":
            dummy["texture_mean"] = dummy.get("texture_mean", 10.0) + 1.0

        df_input = pd.DataFrame([dummy])

        try:
            pred_raw = model.predict(df_input)[0]
            prob_m = get_malignant_probability(model, df_input) 

            pred_label = str(pred_raw)
            if pred_label == "1" or pred_label.lower().startswith("m") or pred_label.lower().startswith("mal"):
                predicted_class = "Malignant"
            elif pred_label == "0" or pred_label.lower().startswith("b") or pred_label.lower().startswith("ben"):
                predicted_class = "Benign"
            else:
                if prob_m is not None:
                    predicted_class = "Malignant" if prob_m >= 0.5 else "Benign"
                else:
                    predicted_class = pred_label

            if prob_m is None:
                # no probability: show deterministic + note
                st.warning("Model tidak menyediakan probabilitas; menampilkan hasil deterministik.")
                if predicted_class == "Malignant":
                    st.error("🔴 **Indikasi: Kemungkinan GANAS (Malignant)**")
                else:
                    st.success("🟢 **Indikasi: Kemungkinan JINAK (Benign)**")
                st.write(f"Deterministic prediction label: **{predicted_class}**")
            else:
                pct = round(float(prob_m) * 100, 1)
                if pct >= 70:
                    tone = "🔴"
                elif pct >= 40:
                    tone = "🟠"
                else:
                    tone = "🟢"

                st.markdown(f"{tone} **Indikasi awal: {predicted_class}**")
                st.write(f"Probabilitas model untuk *ganas*: **{pct}%** (indicative).")

            # Recommendations 
            st.markdown("### Rekomendasi")
            if predicted_class == "Malignant" or (prob_m is not None and prob_m >= 0.5):
                st.warning(
                    "- Terdapat indikasi yang mengarah ke keganasan. Segera lakukan pemeriksaan lanjutan seperti **USG**, **mammogram**, atau **FNA/biopsi**.\n"
                    "- Segera konsultasikan ke **dokter spesialis bedah onkologi** atau onkolog."
                )
            else:
                st.info(
                    "- Indikasi cenderung jinak, tetapi tetap disarankan untuk kontrol dan pemeriksaan lanjutan jika ada perubahan.\n"
                    "- Jika ada pertumbuhan cepat, perubahan bentuk, keluarnya cairan, atau pembengkakan kelenjar, segera periksa ke fasilitas kesehatan."
                )

            with st.expander("Lihat mapping input → fitur (heuristik)"):
                st.write(pd.DataFrame([dummy]).T.rename(columns={0: "value"}))

        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
   
    st.markdown("---")
    chat_ui("chat_patient", chatbot_patient, "💬 Chatbot Informasional Untuk Pasien")
