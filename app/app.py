import os
import sys
import requests
from bs4 import BeautifulSoup
from datetime import datetime

import torch
import torch.nn.functional as F
from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification

# KONFIGURASI PATH & IMPORT LOKAL
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, '..'))

if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import fungsi pembersih teks
try:
    from ml.preprocess import clean_text
except ImportError:
    print("Gagal mengimport clean_text dari ml.preprocess.")
    def clean_text(text): return text 

# Path ke folder model yang sudah di-training
model_dir = os.path.join(root_dir, 'models', 'bert_fake_news')

# INISIALISASI FLASK & LOAD MODEL
app = Flask(__name__)

print(" Loading model BERT")
try:
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval() # Matikan mode training, fokus untuk inferensi (prediksi)
    print("✅ Model berhasil diload!")
except Exception as e:
    print(f"❌ Peringatan: Model gagal diload: {e}")
    tokenizer, model = None, None


# ROUTING & LOGIKA UTAMA
@app.route("/", methods=["GET", "POST"])
def index():
    # Jika user mengirimkan data via form (Klik Analyze)
    if request.method == "POST":
        
        source_format = ""
        news_url = request.form.get("news_url")
        
        # MODE 1: USER INPUT LINK (WEB SCRAPING)
        if news_url:
            source_format = "Web Article"
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                response = requests.get(news_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                news_title = soup.title.string.strip() if soup.title else "Scraped Title"
                
                paragraphs = soup.find_all('p')
                news_text = ' '.join([p.get_text().strip() for p in paragraphs])
                
                if len(news_text) < 20:
                    news_text = "Could not extract sufficient text from this URL."
                    
            except Exception as e:
                news_title = "Error Scraping"
                news_text = f"Failed to retrieve data from URL. Error: {e}"
                
        # MODE 2: USER INPUT TEKS MANUAL
        else:
            source_format = "User Input Text"
            news_title = request.form.get("news_title", "")
            news_text = request.form.get("news_text", "")
    
        # PREPROCESSING & PREDIKSI MODEL
    
        # Teks mentah untuk ditampilkan di UI agar tetap enak dibaca
        raw_input_text = f"{news_title}. {news_text}"
        
        # Teks bersih untuk dimakan oleh Model BERT (sesuai standar saat training)
        cleaned_input_text = clean_text(raw_input_text)
        
        # Ambil tanggal hari ini untuk metadata hasil
        current_date = datetime.now().strftime("%b %d, %Y")

        if model and tokenizer:
            # Ubah teks menjadi token/angka yang dipahami PyTorch
            inputs = tokenizer(
                cleaned_input_text, 
                truncation=True, 
                padding=True, 
                max_length=256, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                
                fake_prob = probs[0][0].item() * 100
                
                if fake_prob >= 50:
                    label = "Hoax"
                    score = fake_prob
                else:
                    label = "Authentic"
                    score = probs[0][1].item() * 100 
        else:
            score = 0.0
            label = "Model Not Loaded"
            
        return render_template(
            "result.html", 
            score=round(score, 2), 
            label=label,
            news_title=news_title,
            news_text=news_text,         
            processed_date=current_date,
            source_format=source_format  
        )

    # Render halaman awal jika request adalah GET (akses web pertama kali)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)