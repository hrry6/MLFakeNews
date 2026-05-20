import pandas as pd
import os
from preprocess import clean_text

# Setup Path 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_data():
    print("Mengola data...")
    
    raw_path = os.path.join(BASE_DIR, 'data', 'raw')
    processed_path = os.path.join(BASE_DIR, 'data', 'processed')
    
    # Validasi folder raw
    if not os.path.exists(raw_path):
        print(f"Folder tidak ditemukan: {raw_path}")
        return

    os.makedirs(processed_path, exist_ok=True)

    # Load Data
    try:
        df_true = pd.read_csv(os.path.join(raw_path, 'True.csv'))
        df_false = pd.read_csv(os.path.join(raw_path, 'Fake.csv'))
        print("Berhasil load True.csv dan Fake.csv")
    except FileNotFoundError as e:
        print(f"File tidak ditemukan: {e.filename}")
        return

    # Labeling (1 = True, 0 = Fake)
    df_true['label'] = 1
    df_false['label'] = 0

    # Gabung Data
    df = pd.concat([df_true, df_false]).reset_index(drop=True)

    # Gabung Title + Text lalu Bersihkan
    print("Membersihkan teks")
    df['content'] = df['title'] + " " + df['text']
    df['content'] = df['content'].apply(clean_text)

    # Ambil kolom yang dibutuhkan
    df = df[['content', 'label']]

    # Hapus Baris Kosong & Duplikat
    initial_len = len(df)
    df = df.dropna(subset=['content'])
    df = df[df['content'].str.strip() != '']
    df = df.drop_duplicates(subset=['content'])
    
    removed_count = initial_len - len(df)
    if removed_count > 0:
        print(f"Menghapus {removed_count} baris duplikat/kosong.")

    # Acak Data & Batasi Sampel 5000 baris
    df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(5000)

    # Simpan Data Processed
    output_file = os.path.join(processed_path, 'train_cleaned.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Data siap pakai tersimpan di: {output_file}")
    print(f"Total baris data akhir: {len(df)}")

if __name__ == "__main__":
    prepare_data()