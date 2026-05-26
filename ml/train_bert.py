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
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'bert_fake_news2')

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
    df = pd.read_csv(DATA_PATH).dropna(subset=['content']).head(5000)
    
    # Split Data (Train & Validation) 
    # 70% untuk belajar (train), 30% untuk ujian (validation)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['content'].astype(str).tolist(), 
        df['label'].tolist(), 
        test_size=0.3, 
        random_state=42
    )

    # Tokenisasi (Ubah Teks Jadi Angka) 
    # Pakai 'cased' agar BERT tetap bisa membaca dan membedakan huruf kapital
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    print("Tokenizing data")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    # Masukkan ke dalam format dataset PyTorch
    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)

    # Load Model Pre-trained
    # num_labels=2 karena kita mengklasifikasikan 2 kelas: Fake (0) dan True (1)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Konfigurasi Training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,            
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",        
        save_strategy="epoch",        
        load_best_model_at_end=True,  
        metric_for_best_model='f1',
        fp16=False,                   
    )

    # Eksekusi Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] 
    )

    trainer.train()

    # Evaluasi Akhir
    print("\nMengevaluasi Model...")
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(val_labels, y_pred))
    
    print("\n--- Classification Report ---")
    # Urutan label: 0 untuk Fake, 1 untuk True
    print(classification_report(val_labels, y_pred, target_names=['Fake', 'True']))

    # Simpan Model Akhir
    print(f"Menyimpan model ke {MODEL_SAVE_PATH}...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Model berhasil disimpan.")

if __name__ == "__main__":
    train()