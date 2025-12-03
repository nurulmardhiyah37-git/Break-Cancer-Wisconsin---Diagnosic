🩺 Breast Cancer Wisconsin – Diagnostic

Machine Learning Classification Project

📘 Deskripsi Proyek
Proyek ini bertujuan membangun model machine learning untuk memprediksi jenis kanker payudara (Malignant atau Benign) menggunakan dataset Breast Cancer Wisconsin (Diagnostic) dari Kaggle. Dataset ini berisi 569 sampel dengan 30 fitur numerik yang menggambarkan morfologi inti sel hasil pemeriksaan Fine Needle Aspiration (FNA).
Proyek ini mengimplementasikan pipeline end-to-end mulai dari data preprocessing, EDA, pemodelan, hingga evaluasi model, serta disiapkan untuk potensi deployment menggunakan Streamlit.

📊 Dataset
Sumber: Kaggle – Breast Cancer Wisconsin (Diagnostic)
Jumlah sampel: 569
Fitur numerik: 30
Label:
1. M → Malignant (ganas)
2. B → Benign (jinak)
   
Keterangan tambahan:
- Dataset tidak memiliki missing value signifikan
- 1 kolom kosong dihapus saat preprocessing

🎯 Tujuan Proyek
Melakukan analisis dan preprocessing terhadap data FNA.
Membangun model klasifikasi untuk memprediksi kanker payudara.
Membandingkan performa beberapa algoritma machine learning.
Menghasilkan pipeline yang terstruktur dan mudah direplikasi.

🧠 Hasil Utama
Model-model klasifikasi berhasil memberikan performa tinggi dengan akurasi di atas 90%, tergantung model dan parameter. Model terbaik dipilih berdasarkan kombinasi accuracy, precision, dan recall, terutama untuk kelas Malignant.
