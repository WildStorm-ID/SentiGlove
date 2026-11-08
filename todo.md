Tolong bangunkan aplikasi web full-stack "SentiGlove" untuk analisis sentimen ulasan produk e-commerce berbasis Python (FastAPI + Streamlit) dan SQLAlchemy ORM, persis sesuai perancangan proposal riset UTS NLP yang dilampirkan.

---

### 1. File & Dataset Input
1. Dataset Ulasan: `tokopedia-product-reviews-2019.csv`
   - Kolom utama: `text` (teks ulasan), `rating` (bintang 1-5), `category`, `product_name`.
   - Konversi Rating ke Label Sentimen:
     * Rating >= 4 -> 'Positif'
     * Rating <= 2 -> 'Negatif'
     * Rating 3 -> Abaikan (Drop) untuk klasifikasi biner.
2. Pretrained Word Embedding: `glove_50dim_wiki.id.case.text.txt` (GloVe Bahasa Indonesia 50 dimensi).

---

### 2. NLP Pipeline & Algoritma Klasifikasi (SentiGlove Engine)
- Preprocessing Teks (Proses 1.0 DFD): Case folding, pembersihan karakter non-alfabet, tokenisasi, normalisasi singkatan, dan penghapusan stopword Bahasa Indonesia.
- Matriks Kookurensi & GloVe Loss (Simulasi/Training Offline - Proses 2.0 & 3.0 DFD): Sediakan modul simulasi kalkulasi matriks kookurensi X (window 5, bobot 1/jarak) dan pembaruan parameter GloVe (w, b) berbasis loss J = Σ f(Xij)·(wi·w̃j + bi + b̃j − log Xij)².
- Vector Averaging Ulasan (50d): Untuk ulasan baru, hitung rata-rata vektor kata (Vector Average) 50 dimensi berdasarkan kamus pretrained GloVe. Jika kata OOV, beri vektor nol 50d.
- Klasifikasi Sentimen (Proses 4.0 DFD): Hitung Centroid Positif (c+) dan Centroid Negatif (c-) dari data latih. Tentukan label ulasan berdasarkan skor Cosine Similarity tertinggi terhadap c+ vs c- (sediakan fallback ke Logistic Regression).

---

### 3. Skema Basis Data Relational (SQLAlchemy ORM - Sesuai ERD 6 Entitas)
Buatkan skema SQLAlchemy ORM (paling cocok SQLite dev lokal) yang persis mencakup 6 entitas dari ERD proposal:
1. PENJUAL (id_penjual, nama_toko)
2. PRODUK (id_produk, id_penjual, nama_produk, kategori)
3. ULASAN (id_ulasan, id_produk, teks, rating_asli, label_asli)
4. MODEL_GLOVE (id_model, dimensi, x_max, alpha, loss_akhir, tanggal_latih)
5. VEKTOR_KATA (id_model, kata, vektor_w, bias_b)
6. HASIL_SENTIMEN (id_hasil, id_ulasan, id_model, skor_pos, skor_neg, label_prediksi)

---

### 4. User Interface Dashboard (Streamlit Web UI)
- Dashboard Penjual: Visualisasi ringkasan produk, proporsi sentimen ulasan (Pie Chart / Bar Chart Positif vs Negatif per produk/kategori).
- Pengujian Ulasan Baru Real-Time: Form input ulasan baru untuk menguji klasifikasi secara live, menampilkan label prediksi, skor Cosine Similarity (c+ vs c-), serta Expander detail teknis vektor 50-dimensi (Vector Average).
- Form Import CSV: Fitur untuk mengunggah file CSV ulasan produk baru dan menyimpan hasilnya ke database relasional.