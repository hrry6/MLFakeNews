import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# KONFIGURASI PATH
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, '..'))

# Path dataset hasil preprocessing
data_path = os.path.join(
    root_dir,
    'data',
    'processed',
    'train_cleaned.csv'
)

# Folder penyimpanan model
model_dir = os.path.join(
    root_dir,
    'models',
    'logistic_regression'
)

# Path file model & vectorizer
model_path = os.path.join(model_dir, 'logreg_model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')


# FUNGSI UTAMA TRAINING
def train():
    print("Memulai proses training Logistic Regression...")

    # CEK DATASET
    if not os.path.exists(data_path):
        print(f"❌ Dataset tidak ditemukan di: {data_path}")
        return

    # LOAD DATASET
    print("Membaca dataset...")

    df = pd.read_csv(data_path)

    # Hapus data kosong
    df = df.dropna(subset=['content'])

    print(f"Total data: {len(df)}")

    # SPLIT DATA TRAIN & VALIDATION
    print("✂️ Membagi data train & validation...")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].astype(str).tolist(),
        df['label'].tolist(),
        test_size=0.3,
        random_state=42,
        stratify=df['label']
    )

    print("Split data selesai")

    # TF-IDF VECTORIZATION
    print("Melakukan TF-IDF Vectorization...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    train_vectors = vectorizer.fit_transform(train_texts)
    val_vectors = vectorizer.transform(val_texts)

    print("Vectorization selesai")

    # INISIALISASI MODEL
    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # TRAIN MODEL
    model.fit(train_vectors, train_labels)

    print("Training selesai!")

    # PREDIKSI VALIDATION
    print("Melakukan evaluasi model...")

    predictions = model.predict(val_vectors)

    # HITUNG METRIK EVALUASI
    accuracy = accuracy_score(val_labels, predictions)
    precision = precision_score(val_labels, predictions)
    recall = recall_score(val_labels, predictions)
    f1 = f1_score(val_labels, predictions)

    # TAMPILKAN HASIL EVALUASI
    print("\n===== HASIL EVALUASI MODEL =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(val_labels, predictions))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(val_labels, predictions))

    # SIMPAN MODEL & VECTORIZER
    print("\nMenyimpan model...")

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("Model berhasil disimpan!")
    print(f"Model      : {model_path}")
    print(f"Vectorizer : {vectorizer_path}")


# MAIN PROGRAM
if __name__ == "__main__":
    train()