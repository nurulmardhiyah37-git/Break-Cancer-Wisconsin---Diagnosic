import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv("data.csv")

# Hapus kolom yang tidak diperlukan
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])

# Hapus id (tidak digunakan untuk training)
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Diagnosis harus ada
if 'diagnosis' not in df.columns:
    raise ValueError("Kolom 'diagnosis' tidak ditemukan!")

# Pisahkan fitur dan label
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
print("Jumlah data setelah preprocessing:", df.shape)
print(df.head())

# 2. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. BUILD MODEL
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": make_pipeline(SimpleImputer(strategy='mean'), GaussianNB()),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}
results = {}
trained_models = {}

# 4. TRAINING DAN EVALUASI
print("\n=== HASIL EVALUASI MASING-MASING MODEL ===\n")
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)

    acc = accuracy_score(y_test, preds)
    # pos_label='M' karena M dianggap positif (tumor ganas)
    prec = precision_score(y_test, preds, pos_label='M')
    rec = recall_score(y_test, preds, pos_label='M')
    f1 = f1_score(y_test, preds, pos_label='M')

    results[name] = acc
    trained_models[name] = mdl

    print(f"Model: {name}")
    print(f"Akurasi: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 40)

# 5. PILIH MODEL TERBAIK
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
print(f"\nMODEL TERBAIK: {best_model_name} (akurasi {results[best_model_name]:.4f})")

# 6. CONFUSION MATRIX MODEL TERBAIK
pred_best = best_model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, pred_best, target_names=['Benign', 'Malignant']))

ConfusionMatrixDisplay.from_predictions(y_test, pred_best)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

# 7. NGESV MODEL TERBAIK
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\nModel terbaik berhasil disimpan sebagai: best_model.pkl")

# 8. SAVE NAMA FITUR UNTUK STREAMLIT
feature_labels = {
    "radius_mean": "Rata-rata Radius Sel",
    "texture_mean": "Rata-rata Tekstur Sel",
    "perimeter_mean": "Rata-rata Keliling Sel",
    "area_mean": "Rata-rata Luas Sel",
    "smoothness_mean": "Rata-rata Kelicinan Permukaan Sel",
    "compactness_mean": "Rata-rata Kekompakan Sel",
    "concavity_mean": "Rata-rata Tingkat Cekungan Sel",
    "concave points_mean": "Rata-rata Titik Cekungan Sel",
    "symmetry_mean": "Rata-rata Simetri Sel",
    "fractal_dimension_mean": "Rata-rata Kompleksitas Bentuk Sel",
    "radius_se": "Standard Error Radius",
    "texture_se": "Standard Error Tekstur",
    "perimeter_se": "Standard Error Keliling",
    "area_se": "Standard Error Luas",
    "smoothness_se": "Standard Error Kelicinan",
    "compactness_se": "Standard Error Kekompakan",
    "concavity_se": "Standard Error Cekungan",
    "concave points_se": "Standard Error Titik Cekungan",
    "symmetry_se": "Standard Error Simetri",
    "fractal_dimension_se": "Standard Error Kompleksitas Bentuk",
    "radius_worst": "Radius Sel Terburuk",
    "texture_worst": "Tekstur Sel Terburuk",
    "perimeter_worst": "Keliling Sel Terburuk",
    "area_worst": "Luas Sel Terburuk",
    "smoothness_worst": "Kelicinan Terburuk",
    "compactness_worst": "Kekompakan Terburuk",
    "concavity_worst": "Cekungan Terburuk",
    "concave points_worst": "Titik Cekungan Terburuk",
    "symmetry_worst": "Simetri Terburuk",
    "fractal_dimension_worst": "Kompleksitas Bentuk Terburuk"
}

# Save labels dan list fitur mentah
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_labels, f)

with open("feature_list.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Nama fitur berhasil disimpan untuk Streamlit!")
