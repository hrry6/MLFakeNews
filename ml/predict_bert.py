import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from preprocess import clean_text

# Setup Path (Arahkan ke folder models di root project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bert_fake_news')

# Load Tokenizer & Model
print("Loading model...")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval() # Set mode evaluasi
except Exception as e:
    print(f"Gagal meload model. Pastikan path benar dan train.py sudah dijalankan. Error: {e}")
    exit()

def prediksi_berita(judul, isi_berita):
    """Fungsi untuk memprediksi teks berita berdasarkan judul dan isi utuh."""
    
    # 1. Gabungkan judul dan isi persis seperti saat fase training (di prepare_data.py)
    teks_gabungan = f"{judul} {isi_berita}"
    
    # 2. Bersihkan teks agar format sama dengan data training
    teks_bersih = clean_text(teks_gabungan)
    
    # 3. Tokenisasi
    inputs = tokenizer(teks_bersih, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    # 4. Prediksi
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 5. Ambil label dan hitung probabilitas (Index 0 = Fake, Index 1 = True)
    prediksi = torch.argmax(outputs.logits, dim=-1).item()
    probabilitas = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    label = "ASLI (True)" if prediksi == 1 else "HOAX/PALSU (Fake)"
    
    # Tampilkan hasil
    print(f"\nJudul: {judul}")
    print(f"Isi Singkat: {isi_berita[:100]}...")
    print(f"Hasil Prediksi: {label} (Keyakinan: {probabilitas[prediksi].item() * 100:.2f}%)")
    print("-" * 50)

if __name__ == "__main__":
    
    # TEST 1: Berita palsu bergaya sensasional/konspirasi (Tanpa sumber jelas)
    judul_fake = "BREAKING: Secret Military Tribunal Arrests Former President in Midnight Raid!"
    isi_fake = "In a stunning turn of events, the former president was reportedly arrested late last night by a secret military tribunal. Sources claim the raid uncovered evidence of massive voter fraud and alien communication devices. Mainstream media is completely covering this up. Wake up citizens, the truth is being hidden from you!"
    prediksi_berita(judul_fake, isi_fake)

    # TEST 2: Berita asli bergaya jurnalistik formal (Sengaja dikasih embel-embel Reuters untuk menguji regex clean_text)
    judul_true = "Senate narrowly passes sweeping tax overhaul bill"
    isi_true = "WASHINGTON (Reuters) - The Senate narrowly passed a sweeping tax overhaul on Saturday, moving the administration a major step closer to their goal of slashing taxes for businesses while offering everyday citizens a mixed bag of changes. The 51-49 vote came after marathon debates extending into the early hours of the morning."
    prediksi_berita(judul_true, isi_true)