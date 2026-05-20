import re

def clean_text(text):
    """
    Fungsi untuk membersihkan teks sebelum dimasukkan ke model BERT.
    Tanda baca, angka, dan huruf kapital dipertahankan karena BERT membutuhkannya untuk konteks.
    """
    # Validasi tipe data
    if not isinstance(text, str):
        return ""
    
    # Hapus Pola Reuters/Agensi Berita (Case-Insensitive untuk mencegah kebocoran label)
    text = re.sub(r'(?i)^.*?\(reuters\)\s*-\s*', '', text)
    text = re.sub(r'(?i)^.*?reuters\s*-\s*', '', text)
    
    # Hapus URL/Link Website
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Hapus HTML Tags (Sisa web scraping)
    text = re.sub(r'<.*?>', '', text)
    
    # Hapus spasi kosong di awal dan akhir kalimat
    text = text.strip()
    
    # Satukan spasi ganda di tengah kalimat
    text = re.sub(r'\s+', ' ', text)
    
    return text

if __name__ == "__main__":
    # Area Testing
    sample_true = "WASHINGTON (Reuters) - The government passed a new economic tax law today at 15:00."
    sample_fake = "BREAKING NEWS: Aliens have landed near https://nasa.gov right now!!!"
    
    print("--- HASIL TESTING CLEAN_TEXT ---")
    print(f"Sebelum (True) : {sample_true}")
    print(f"Sesudah (True) : {clean_text(sample_true)}\n")
    
    print(f"Sebelum (Fake) : {sample_fake}")
    print(f"Sesudah (Fake) : {clean_text(sample_fake)}")