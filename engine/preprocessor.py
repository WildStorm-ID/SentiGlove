"""
Preproses Ulasan Teks Bahasa Indonesia (Proses 1.0 DFD)
Mencakup:
- Case Folding (lowercase)
- Pembersihan karakter non-alfabet (regex cleaning)
- Tokenisasi
- Normalisasi singkatan / slang e-commerce Bahasa Indonesia
- Penghapusan stopword (dengan mempertahankan penanda negasi penting: tidak, bukan, kurang, belum, jangan)
"""

import re
from typing import List, Set, Dict

# Kamus normalisasi kata gaul/singkatan umum pada e-commerce Indonesia
SLANG_DICTIONARY: Dict[str, str] = {
    # Sentimen Positif & Penguat
    "bgs": "bagus",
    "bgus": "bagus",
    "bguss": "bagus",
    "mantul": "mantap",
    "mantab": "mantap",
    "mantapp": "mantap",
    "mantabb": "mantap",
    "top": "bagus",
    "ok": "bagus",
    "oke": "bagus",
    "okee": "bagus",
    "okeeh": "bagus",
    "sip": "bagus",
    "sipp": "bagus",
    "jos": "bagus",
    "joss": "bagus",
    "josss": "bagus",
    "juara": "bagus",
    "keren": "bagus",
    "apik": "bagus",
    "bgt": "banget",
    "bgtt": "banget",
    "bangett": "banget",
    "buanget": "banget",
    "recomended": "rekomendasi",
    "recommended": "rekomendasi",
    "rekomen": "rekomendasi",
    "rekomended": "rekomendasi",
    "recsel": "rekomendasi",
    "ori": "asli",
    "original": "asli",
    "origina": "asli",
    "orisinil": "asli",
    "rapih": "rapi",
    "cpt": "cepat",
    "cepet": "cepat",
    "cepetan": "cepat",
    "puas": "puas",
    "puass": "puas",
    "seneng": "senang",
    "suka": "suka",
    "sukaa": "suka",
    "sukses": "sukses",
    "terbaik": "terbaik",
    "the best": "terbaik",
    
    # Sentimen Negatif & Keluhan
    "ancur": "hancur",
    "hancurr": "hancur",
    "rusakk": "rusak",
    "pecahh": "pecah",
    "patahh": "patah",
    "cacatt": "cacat",
    "kecewaa": "kecewa",
    "kcewa": "kecewa",
    "kecewa": "kecewa",
    "nyesel": "menyesal",
    "menyesall": "menyesal",
    "kapok": "kapok",
    "rugi": "rugi",
    "parahh": "parah",
    "jelekk": "jelek",
    "jlk": "jelek",
    "lemot": "lambat",
    "lola": "lambat",
    "lelet": "lambat",
    "kw": "tiruan",
    "palsu": "palsu",
    "bohong": "bohong",
    "tipu": "tipu",
    
    # Negasi (sangat krusial untuk sentimen)
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "engga": "tidak",
    "enggak": "tidak",
    "kagak": "tidak",
    "bkn": "bukan",
    "g": "tidak",
    "ndak": "tidak",
    "krg": "kurang",
    "blm": "belum",
    "blom": "belum",
    "belom": "belum",
    "jgn": "jangan",
    
    # Kata Benda & Transaksi E-commerce
    "brg": "barang",
    "brng": "barang",
    "barangg": "barang",
    "bary": "barang",
    "item": "barang",
    "produk": "produk",
    "kwalitas": "kualitas",
    "kualitass": "kualitas",
    "packing": "kemasan",
    "paking": "kemasan",
    "bungkus": "kemasan",
    "kemasann": "kemasan",
    "seller": "penjual",
    "saler": "penjual",
    "toko": "toko",
    "tokonya": "toko",
    "admin": "admin",
    "kurir": "kurir",
    "ekspedisi": "ekspedisi",
    "pengiriman": "pengiriman",
    "kirim": "kirim",
    "ongkir": "ongkir",
    "respon": "respons",
    "responnya": "respons",
    "responsnya": "respons",
    "harganya": "harga",
    "hrg": "harga",
    
    # Kata Hubung & Ganti
    "yg": "yang",
    "yng": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "sy": "saya",
    "ak": "aku",
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "lu": "kamu",
    "km": "kamu",
    "dr": "dari",
    "dri": "dari",
    "utk": "untuk",
    "untk": "untuk",
    "kpd": "kepada",
    "krn": "karena",
    "karna": "karena",
    "d": "di",
    "jd": "jadi",
    "jdi": "jadi",
    "tp": "tapi",
    "tpi": "tapi",
    "sdh": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "dah": "sudah",
    "sampe": "sampai",
    "ampe": "sampai",
    "bs": "bisa",
    "bisaa": "bisa",
    "aja": "saja",
    "ajah": "saja",
    "aj": "saja",
    "sj": "saja",
    "trs": "terus",
    "lg": "lagi",
    "lgi": "lagi",
    "bener": "benar",
    "bnr": "benar",
    "beneran": "benar",
    "bngt": "banget",
    "gitu": "begitu",
    "gt": "begitu",
    "gini": "begini",
    "gn": "begini",
    "sm": "sama",
    "smp": "sampai",
    "dpt": "dapat",
    "dapet": "dapat",
    "dptnya": "dapat",
    "kasih": "terima kasih",
    "makasih": "terima kasih",
    "thx": "terima kasih",
    "tq": "terima kasih",
    "tengkyu": "terima kasih",
    "thanks": "terima kasih"
}

# Stopwords standar Bahasa Indonesia yang tidak mengubah polaritas sentimen
STANDARD_STOPWORDS: Set[str] = {
    "yang", "di", "dan", "dari", "ini", "itu", "ke", "pada", "adalah",
    "yaitu", "oleh", "untuk", "dengan", "saya", "aku", "kamu", "dia",
    "mereka", "kami", "kita", "ia", "nya", "si", "sang", "para", "hal",
    "secara", "seperti", "sebagai", "bagi", "tentang", "maka", "atau",
    "serta", "serta", "jika", "kalau", "bila", "supaya", "agar",
    "sehingga", "karena", "sebab", "oleh", "karena", "terhadap", "dalam",
    "antara", "setelah", "sesudah", "sebelum", "ketika", "saat", "waktu",
    "hingga", "sampai", "tersebut", "terhadap", "pada", "suatu", "bisa",
    "dapat", "sudah", "telah", "sedang", "lagi", "terus", "jadi", "saja"
}

# Kata sentimen & negasi yang WAJIB dipertahankan (jangan dihapus oleh stopword filtering)
SENTIMENT_PRESERVED_WORDS: Set[str] = {
    "tidak", "bukan", "kurang", "belum", "jangan", "tak", "tanpa",
    "bagus", "jelek", "kecewa", "puas", "cepat", "lambat", "rusak", "cacat",
    "pecah", "asli", "palsu", "tiruan", "mantap", "ramah", "rapi", "suka"
}


def case_folding(text: str) -> str:
    """Mengubah seluruh huruf menjadi huruf kecil."""
    return text.lower() if isinstance(text, str) else ""


def clean_text(text: str) -> str:
    """
    Membersihkan teks ulasan:
    - Menghapus URL/link web
    - Menghapus mention (@user) dan hashtag (#tag)
    - Menghapus karakter non-alfabet (angka, tanda baca, simbol, emoji)
    - Merapikan spasi berlebih
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    t = text.lower()
    # Hapus URL
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    # Hapus mention dan hashtag
    t = re.sub(r"[@#]\w+", " ", t)
    # Ganti karakter non-alfabet dengan spasi
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    # Gabungkan spasi berlebih
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    """Memecah teks bersih menjadi daftar token (kata)."""
    return [token for token in text.split() if token]


def normalize_slang(tokens: List[str]) -> List[str]:
    """Menormalisasi kata singkatan / slang sesuai kamus bahasa gaul."""
    normalized = []
    for token in tokens:
        # Periksa apakah ada pemetaan singkatan
        norm_val = SLANG_DICTIONARY.get(token, token)
        # Jika hasil pemetaan mengandung spasi (misal "terima kasih"), pecah kembali
        for sub_tok in norm_val.split():
            normalized.append(sub_tok)
    return normalized


def remove_stopwords(tokens: List[str], preserve_sentiment: bool = True) -> List[str]:
    """
    Menghapus stopword Bahasa Indonesia.
    Jika preserve_sentiment=True, kata negasi dan sentimen tidak akan dibuang.
    """
    filtered = []
    for token in tokens:
        if preserve_sentiment and token in SENTIMENT_PRESERVED_WORDS:
            filtered.append(token)
            continue
        if token not in STANDARD_STOPWORDS and len(token) > 1:
            filtered.append(token)
    return filtered


def preprocess_review(text: str, remove_stop: bool = True) -> List[str]:
    """
    Pipeline lengkap Praproses Ulasan (Proses 1.0 DFD):
    1. Clean text (Case folding + Regex pembersihan)
    2. Tokenize
    3. Normalisasi slang / singkatan
    4. Penghapusan stopword (opsional, mempertahankan kata negasi)
    """
    cleaned = clean_text(text)
    raw_tokens = tokenize(cleaned)
    normalized_tokens = normalize_slang(raw_tokens)
    if remove_stop:
        final_tokens = remove_stopwords(normalized_tokens, preserve_sentiment=True)
    else:
        final_tokens = normalized_tokens
    return final_tokens

