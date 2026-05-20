# Veritas — Machine Learning Fake News Detection System

Veritas adalah platform deteksi integritas informasi digital (*fake news* / hoax) berbasis web yang menggunakan model NLP *state-of-the-art* **BERT (`bert-base-cased`)**. Aplikasi ini dirancang dengan arsitektur bersih (*monorepo*) yang memisahkan ekosistem *Machine Learning pipeline* dengan aplikasi web Flask.

## Fitur Utama
- **Analyze Text**: Deteksi langsung dengan memasukkan judul dan isi artikel berita secara manual.
- **Paste Link (Web Scraping)**: Ekstraksi otomatis konten artikel dan judul dari URL berita menggunakan *BeautifulSoup4*.
- **Context-Aware Preprocessing**: Pembersihan teks tingkat lanjut yang mempertahankan tanda baca, angka, dan kapitalisasi untuk menjaga akurasi pemahaman konteks oleh model BERT.

---

## 📂 Struktur Proyek (Monorepo)

```

```text
README.md generated successfully.

```text
MLFakeNewsTemplate/
├── app/                  # Aplikasi Web & User Interface (Flask)
│   ├── app.py            # Main server Flask
│   ├── templates/        # File HTML (index.html, result.html)
│   └── static/           # Asset statis (CSS, JS)
├── ml/                   # Machine Learning Pipeline
│   ├── preprocess.py     # Logika pembersihan teks teks (BERT-friendly)
│   ├── prepare_data.py   # Penggabungan data, deduplikasi, & sampling
│   ├── train.py          # Script finetuning BERT dengan PyTorch & HuggingFace Trainer
│   └── predict.py        # Script CLI untuk testing model secara lokal
├── data/                 # Penyimpanan Data Dataset (ISOT)
│   ├── raw/              # True.csv dan Fake.csv asli
│   └── processed/        # train_cleaned.csv hasil preprocessing
├── models/               # Tempat penyimpanan akhir model hasil training
│   └── bert_fake_news/   # Bobot model & konfigurasi tokenizer (.safetensors)
├── Dockerfile            # Blueprint containerisasi aplikasi
├── docker-compose.yml    # Orkestrasi container dan volume mapping
└── requirements.txt      # Daftar dependencies Python

```

---

## Alur Training & Model

### 1. Spesifikasi Model

* **Arsitektur**: `bert-base-cased` (HuggingFace Transformers).
* **Alasan**: Berbeda dengan *uncased*, versi *cased* mempertahankan huruf kapital yang sangat krusial dalam mendeteksi gaya bahasa *clickbait* atau hoax. Tanda baca dan angka juga tidak dihapus agar mekanis *Attention* BERT memahami struktur sintaksis kalimat secara utuh.

### 2. Pipeline Pengolahan Data & Training

1. **Pembersihan Data Leakage (`ml/preprocess.py`)**: Menghapus pola agensi berita kaku seperti `(Reuters) - ` secara *case-insensitive* untuk mencegah model mengalami *overfitting* atau menghafal *watermark* sumber teks asli.
2. **Deduplikasi (`ml/prepare_data.py`)**: Menghapus data duplikat dan baris kosong setelah pembersihan teks untuk mencegah kebocoran informasi (*data leakage*) antara data training dan validasi.
3. **Finetuning (`ml/train.py`)**: Melakukan training menggunakan *HuggingFace Trainer API* dengan kriteria optimasi berbasis *F1-Score*. Model terbaik di akhir epoch otomatis disimpan ke direktori `models/bert_fake_news/`.

---

## Cara Setup & Menjalankan Menggunakan Docker

Pastikan Docker dan Docker Compose sudah terinstal di komputer Anda.

### 1. Build dan Jalankan Container

Buka terminal pada *root directory* proyek (`MLFakeNewsTemplate/`), lalu jalankan perintah berikut:

```bash
docker-compose up --build

```

> **Catatan**: Flag `--build` digunakan untuk memastikan Docker menginstal ulang semua dependencies yang terdaftar di `requirements.txt` ke dalam *image* (termasuk instalasi PyTorch dan Transformers).

### 2. Akses Aplikasi

Setelah log container menampilkan pesan sukses (`Running on http://0.0.0.0:5000`), buka browser Anda dan akses:

```text
http://localhost:5000

```

### 3. Mematikan Aplikasi

Untuk menghentikan *container* yang sedang berjalan, tekan `CTRL + C` pada terminal, atau jalankan perintah berikut di tab terminal baru:

```bash
docker-compose down

```