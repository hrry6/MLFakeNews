import os
import sys
import requests
import joblib
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

try:
    from ml.preprocess import clean_text
except ImportError:
    print("Gagal mengimport clean_text dari ml.preprocess.")
    def clean_text(text): return text 

# Path ke folder model BERT
bert_dir = os.path.join(root_dir, 'models', 'bert_fake_news')

# Path ke folder model Logistic Regression
logreg_dir = os.path.join(root_dir, 'models', 'logistic_regression')
logreg_model_path = os.path.join(logreg_dir, 'logreg_model.pkl')
vectorizer_path = os.path.join(logreg_dir, 'tfidf_vectorizer.pkl')

# INISIALISASI FLASK & LOAD MODEL
app = Flask(__name__)

print("Loading model BERT...")
try:
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    bert_model = BertForSequenceClassification.from_pretrained(bert_dir)
    bert_model.eval()
    print("Model BERT berhasil diload!")
except Exception as e:
    print(f"Peringatan: Model BERT gagal diload: {e}")
    tokenizer, bert_model = None, None

print("Loading model Logistic Regression...")
try:
    logreg_model = joblib.load(logreg_model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model Logistic Regression berhasil diload!")
except Exception as e:
    print(f"Peringatan: Model Logistic Regression gagal diload: {e}")
    logreg_model, vectorizer = None, None

# ROUTING & LOGIKA UTAMA
@app.route("/", methods=["GET", "POST"])
def index():
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
        raw_input_text = f"{news_title}. {news_text}"
        cleaned_input_text = clean_text(raw_input_text)
        current_date = datetime.now().strftime("%b %d, %Y")

        # PREDIKSI MENGGUNAKAN BERT
        if bert_model and tokenizer:
            inputs = tokenizer(
                cleaned_input_text, 
                truncation=True, 
                padding=True, 
                max_length=256, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                
                fake_prob = probs[0][0].item() * 100
                
                if fake_prob >= 50:
                    bert_label = "Hoax"
                    bert_score = fake_prob
                else:
                    bert_label = "Authentic"
                    bert_score = probs[0][1].item() * 100 
        else:
            bert_score = 0.0
            bert_label = "BERT Not Loaded"

        if logreg_model and vectorizer:
            teks_vector = vectorizer.transform([cleaned_input_text])
            logreg_pred = logreg_model.predict(teks_vector)[0]
            logreg_probs = logreg_model.predict_proba(teks_vector)[0]

            if logreg_pred == 1:
                logreg_label = "Authentic"
                logreg_score = logreg_probs[1] * 100
            else:
                logreg_label = "Hoax"
                logreg_score = logreg_probs[0] * 100
        else:
            logreg_score = 0.0
            logreg_label = "LogReg Not Loaded"
            
        # KIRIM DATA KE HALAMAN HASIL
        return render_template(
            "result.html", 
            bert_score=round(bert_score, 2),
            bert_label=bert_label,
            logreg_score=round(logreg_score, 2),
            logreg_label=logreg_label,
            news_title=news_title,
            news_text=news_text,         
            processed_date=current_date,
            source_format=source_format  
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)