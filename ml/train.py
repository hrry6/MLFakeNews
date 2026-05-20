import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

# SETUP PATH LOKASI FILE
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'train_cleaned.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'bert_fake_news')

# FUNGSI METRIK EVALUASI
def compute_metrics(pred):
    """Menghitung metrik F1-Score dan Accuracy saat evaluasi model."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary'),
    }

# KELAS DATASET CUSTOM
class NewsDataset(torch.utils.data.Dataset):
    """Membungkus data teks dan label agar bisa diproses oleh PyTorch/Trainer."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# FUNGSI UTAMA TRAINING
def train():
    print("Memulai proses training BERT...")
   
    #  Load & Cek Data 
    if not os.path.exists(DATA_PATH):
        print(f"Data tidak ditemukan di {DATA_PATH}. Jalankan prepare_data.py dulu!")
        return
    
    # Baca data dan buang baris yang kosong
    df = pd.read_csv(DATA_PATH).dropna(subset=['content'])
    
    # Split Data (Train & Validation) 
    # 70% untuk belajar (train), 30% untuk ujian (validation)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].astype(str).tolist(), 
        df['label'].tolist(), 
        test_size=0.3, 
        random_state=42
    )

    #  Tokenisasi (Ubah Teks Jadi Angka) 
    # Pakai 'cased' agar BERT tetap bisa membaca dan membedakan huruf kapital
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    print("Tokenizing data")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    # Masukkan ke dalam format dataset PyTorch
    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_