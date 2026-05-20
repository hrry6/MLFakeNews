import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from preprocess import clean_text

# Setup Path (Arahkan ke folder models di root project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bert_fake_news')

# Load Tokenizer & Model
print("⏳ Loading model...")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval() # Set mode evaluasi
except Exception as e:
    print(f"❌ Gagal meload model. Pastikan path benar dan train.py sudah dijalankan. Error: {e}")
    exit()

def prediksi_berita(teks):
    """Fungsi untuk memprediksi teks berita tunggal."""
    # Bersihkan teks agar format sama dengan data training
    teks_bersih = clean_text(teks)
    
    # Tokenisasi
    inputs = tokenizer(teks_bersih, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    # Prediksi
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Ambil label dan hitung probabilitas
    prediksi = torch.argmax(outputs.logits, dim=-1).item()
    probabilitas = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    label = "ASLI (True)" if prediksi == 1 else "PALSU (Fake)"
    
    # Tampilkan hasil
    print(f"\n📰 Teks Asli: {teks}")
    print(f"🤖 Hasil Prediksi: {label} (Keyakinan: {probabilitas[prediksi].item() * 100:.2f}%)")

if __name__ == "__main__":
    # TEST 1: Berita palsu, sengaja dikasih embel-embel Reuters untuk menguji bypass
    prediksi_berita("WASHINGTON (Reuters) - Aliens have landed on Earth and they are currently eating fried rice in Jakarta.")

    # TEST 2: Berita asli biasa tanpa embel-embel agensi berita
    prediksi_berita("The president announced a new economic policy during the conference today to reduce tax rates.")