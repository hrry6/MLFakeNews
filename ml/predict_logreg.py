import os
import joblib

from preprocess import clean_text

# SETUP PATH
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, '..'))

# Path folder model Logistic Regression
model_dir = os.path.join(
    root_dir,
    'models',
    'logistic_regression'
)

# Path file model & vectorizer
model_path = os.path.join(model_dir, 'logreg_model.pkl')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

# LOAD MODEL & VECTORIZER
print("Loading model Logistic Regression...")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model berhasil diload!")

except Exception as e:
    print(f"Gagal meload model: {e}")
    exit()


# FUNGSI PREDIKSI
def prediksi_berita(judul, isi_berita):
    """Fungsi untuk memprediksi teks berita berdasarkan judul dan isi utuh."""

    # 1. Gabungkan judul dan isi persis seperti format training
    teks_gabungan = f"{judul} {isi_berita}"

    # 2. Bersihkan teks
    teks_bersih = clean_text(teks_gabungan)

    # 3. Ubah teks menjadi vector TF-IDF
    teks_vector = vectorizer.transform([teks_bersih])

    # 4. Prediksi label
    prediksi = model.predict(teks_vector)[0]

    # 5. Ambil probabilitas
    probabilitas = model.predict_proba(teks_vector)[0]

    # Tentukan label (Asumsi: 1 = True, 0 = Fake sesuai prepare_data.py)
    label = "ASLI (True)" if prediksi == 1 else "HOAX/PALSU (Fake)"

    # Confidence score
    confidence = probabilitas[prediksi] * 100

    # Tampilkan hasil (Format disamakan dengan predict_bert.py)
    print(f"\nJudul: {judul}")
    print(f"Isi Singkat: {isi_berita[:100]}...")
    print(f"Hasil Prediksi: {label} (Keyakinan: {confidence:.2f}%)")
    
    # Detail probabilitas khusus LogReg (opsional, bagus buat debugging)
    print(f"   [Detail] Probabilitas Fake: {probabilitas[0] * 100:.2f}% | True: {probabilitas[1] * 100:.2f}%")
    print("-" * 60)


# MAIN PROGRAM
if __name__ == "__main__":

    # TEST 1: Berita palsu bergaya sensasional/konspirasi
    judul_fake = "BREAKING: Secret Military Tribunal Arrests Former President in Midnight Raid!"
    isi_fake = "In a stunning turn of events, the former president was reportedly arrested late last night by a secret military tribunal. Sources claim the raid uncovered evidence of massive voter fraud and alien communication devices. Mainstream media is completely covering this up. Wake up citizens, the truth is being hidden from you!"
    prediksi_berita(judul_fake, isi_fake)

    # TEST 2: Berita asli bergaya jurnalistik formal (Sengaja dikasih embel-embel Reuters)
    judul_true = "Senate narrowly passes sweeping tax overhaul bill"
    isi_true = "WASHINGTON (Reuters) - The Senate narrowly passed a sweeping tax overhaul on Saturday, moving the administration a major step closer to their goal of slashing taxes for businesses while offering everyday citizens a mixed bag of changes. The 51-49 vote came after marathon debates extending into the early hours of the morning."
    prediksi_berita(judul_true, isi_true)