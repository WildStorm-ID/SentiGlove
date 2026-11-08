# 🧤 SentiGlove: Analisis Sentimen Ulasan E-Commerce Berbasis Word Embedding GloVe

**Tugas UTS Natural Language Processing (NLP) · Studi Kasus Word Embedding: GloVe**  
Aplikasi Full-Stack Web Analisis Sentimen Ulasan Produk Marketplace dengan **GloVe 50-Dimensi Bahasa Indonesia**, **SQLAlchemy ORM (SQLite)**, **FastAPI**, dan **Streamlit**.

---

## 📌 1. Pendahuluan & Latar Belakang

Penjual di marketplace (*Tokopedia, Shopee*) menerima ribuan ulasan per produk sehingga membaca satu per satu sangat tidak efisien. Ragam ungkapan sentimen informal bahasa Indonesia (*“bgs bgt”*, *“mantul”*, *“jelek parah”*, *“gak sesuai”*) menyulitkan daftar kata manual.

**GloVe (Global Vectors for Word Representation - Pennington dkk., 2014)** membangun vektor kata dari statistik kookurensi global seluruh korpus:
$$J = \sum_{i,j=1}^{V} f(X_{ij}) \cdot \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
dengan fungsi bobot pembatas:
$$f(x) = \begin{cases} \left( \frac{x}{x_{\max}} \right)^\alpha & \text{jika } x < x_{\max} \\ 1 & \text{lainnya} \end{cases}$$

Kata dengan konteks sentimen serupa menempati daerah berdekatan dalam ruang vektor 50-dimensi.

---

## 🏗️ 2. Arsitektur & Perancangan Sistem

### 2.1 Aliran Data (DFD Level 0 & Level 1)
- **DFD Level 0 (Diagram Konteks):** 
  - **Penjual:** Mengirim ulasan baru $\rightarrow$ Menerima label sentimen ulasan.
  - **Admin Data:** Memasok korpus data ulasan $\rightarrow$ Menerima laporan model.
- **DFD Level 1:**
  - **Proses 1.0 (Praproses Ulasan):** Case folding, pembersihan non-alfabet, tokenisasi, normalisasi singkatan e-commerce, dan penghapusan stopword (mempertahankan kata negasi: *tidak, bukan, kurang, belum, jangan*).
  - **Proses 2.0 (Matriks Kookurensi):** Menghitung matriks kookurensi $X(i, j)$ dengan jendela konteks simetris (*window* 5 atau 1) dan pembobotan jarak $1/\text{jarak}$.
  - **Proses 3.0 (Latih GloVe):** Mengoptimasi parameter $(w, \tilde{w}, b, \tilde{b})$ berbasis fungsi loss kuadrat terbobot dengan SGD / AdaGrad.
  - **Proses 4.0 (Klasifikasi Sentimen):** Menghitung **Vector Averaging (50d)** ulasan, lalu menentukan label berdasarkan **Cosine Similarity** tertinggi ke pusat kelas positif ($c^+$) vs negatif ($c^-$), dengan fallback Regresi Logistik.

### 2.2 Skema Basis Data Relasional (ERD SQLAlchemy ORM)
Aplikasi mengimplementasikan secara presisi 6 entitas dari diagram ERD proposal:
1. `PENJUAL` (`id_penjual` PK, `nama_toko`)
2. `PRODUK` (`id_produk` PK, `id_penjual` FK, `nama_produk`, `kategori`)
3. `ULASAN` (`id_ulasan` PK, `id_produk` FK, `teks`, `rating_asli`, `label_asli`, `created_at`)
4. `MODEL_GLOVE` (`id_model` PK, `dimensi`, `x_max`, `alpha`, `loss_akhir`, `tanggal_latih`, `keterangan`)
5. `VEKTOR_KATA` (`id` PK, `id_model` FK, `kata`, `vektor_w`, `bias_b`)
6. `KOOKURENSI` (`id` PK, `id_model` FK, `kata_i`, `kata_j`, `jumlah`)
7. `HASIL_SENTIMEN` (`id_hasil` PK, `id_ulasan` FK, `id_model` FK, `skor_pos`, `skor_neg`, `label_prediksi`, `created_at`)

---

## 🧮 3. Simulasi Perhitungan Matematis (Sesuai Bab 4 Proposal)

Aplikasi memiliki modul kalkulator interaktif yang mereplikasi secara eksak perhitungan Bab 4:
1. **4.1 Matriks Kookurensi $X$ (Tabel 3):** Korpus 3 ulasan (*“barang bagus sekali”*, *“barang jelek sekali”*, *“pengiriman cepat barang bagus”*), window=1.
2. **4.2 Fungsi Bobot, Loss, dan Pembaruan SGD (Tabel 4):**
   - $X_{ij}=2 \implies \log(2) = 0.6931$
   - $f(2) = (2 / 10)^{0.75} = 0.2991$
   - $\text{pred} = w \cdot \tilde{w} + b + \tilde{b} = 0.2500$
   - $d = \text{pred} - \log X = -0.4431$
   - $J = f \cdot d^2 = 0.0587$
   - Gradien $g = 2 f d = -0.2651$
   - Pembaruan SGD ($\eta = 0.1$): $w \leftarrow [0.3053, 0.1106]$, $\tilde{w} \leftarrow [0.2080, 0.4027]$, $b \leftarrow 0.1265$, $\tilde{b} \leftarrow 0.0765$
   - Evaluasi Ulang: $d_{\text{baru}} = -0.3821$, $J_{\text{baru}} = 0.0437$ (**Loss terbukti turun dari 0.0587 ke 0.0437!**).
3. **4.3 Klasifikasi Sentimen Ulasan Baru (Tabel 5):**
   - Centroid $c^+ = [0.525, 0.400]$, $c^- = [-0.550, -0.450]$
   - Ulasan *“barang cepat bagus”* $\implies \cos(r, c^+) = 0.998, \cos(r, c^-) = -0.995 \implies$ **Positif**
   - Ulasan *“barang jelek sekali”* $\implies \cos(r, c^+) = -0.898, \cos(r, c^-) = 0.913 \implies$ **Negatif**

---

## 📁 4. Struktur Direktori Proyek

```
SentiGlove/
├── api/                        # Backend REST API (FastAPI)
│   ├── main.py                 # Server FastAPI, CORS, app mounting
│   ├── routes.py               # Endpoints (/health, /predict, /simulate, /products, /reviews)
│   └── schemas.py              # Pydantic Schemas untuk request & response
├── database/                   # Layer Basis Data Relasional (SQLAlchemy ORM)
│   ├── db.py                   # Engine SQLite & Session Maker
│   ├── models.py               # 6 Entitas ERD Proposal (Penjual, Produk, Ulasan, ModelGlove, dll)
│   └── init_db.py              # Inisialisasi skema & seeding dari Tokopedia reviews
├── engine/                     # SentiGlove Core NLP Engine
│   ├── preprocessor.py         # Case folding, slang normalization, stopword filtering
│   ├── glove_loader.py         # Pemuatan GloVe 50D binary cache (.npz) & Vector Averaging
│   ├── cooccurrence.py         # Pembangunan Matriks Kookurensi X (window & distance weighting)
│   ├── simulation.py           # Replikasi matematis perhitungan GloVe Bab 4
│   └── classifier.py           # Centroid Classifier (c+ vs c-) & Fallback Regresi Logistik
├── frontend/                   # UI Dashboard Interaktif (Streamlit)
│   ├── app.py                  # Aplikasi utama Streamlit dengan 5 Modul Navigasi
│   └── components.py           # Visualisasi chart Altair (Donut, Bar, 50D Vector) & styling
├── model and dataset/          # File dataset & word embedding
│   ├── tokopedia-product-reviews-2019.csv
│   ├── glove_50dim_wiki.id.case.text.txt
│   ├── glove_50dim.npz         # Binary cache untuk loading cepat (< 1.5 detik)
│   └── senti_glove_classifier.pkl
├── tests/
│   └── test_engine.py          # Unit tests mencakup 6 pengujian validasi menyeluruh
├── requirements.txt            # Dependensi paket Python
├── run.py                      # Skrip peluncur all-in-one
└── README.md
```

---

## 🚀 5. Cara Menjalankan Aplikasi

### 1. Prasyarat & Instalasi Dependensi
Pastikan Python 3.10+ telah terinstal, lalu pasang dependensi:
```bash
pip install -r requirements.txt
```

### 2. Inisialisasi Basis Data (Jika Belum Ada)
```bash
python run.py --init-db
```

### 3. Menjalankan Dashboard Web Streamlit (Rekomendasi)
```bash
python run.py
```
Buka browser pada URL: **`http://localhost:8501`**

### 4. Menjalankan REST API FastAPI
```bash
python run.py --api
```
- Endpoint REST API: **`http://127.0.0.1:8000`**
- Dokumentasi Interaktif Swagger UI: **`http://127.0.0.1:8000/docs`**

### 5. Menjalankan Keduanya Bersamaan (FastAPI + Streamlit)
```bash
python run.py --all
```

### 6. Menjalankan Unit Tests
```bash
python run.py --test
```

---

## 🌟 6. Fitur Utama Dashboard

1. **Dashboard Penjual:**
   - Metrik KPI Toko: Total Toko, Total Produk, Total Ulasan, Rasio Sentimen Positif, Rating Rata-rata.
   - Filter dinamis berdasarkan Kategori, Toko/Penjual, dan Produk Spesifik.
   - Donut Chart Sentimen Positif vs Negatif.
   - Bar Chart Distribusi Rating Asli (Bintang 1-5).
   - Bar Chart Komparasi Sentimen per Kategori Produk.
   - Tabel Ulasan Terkini lengkap dengan skor Cosine Similarity.

2. **Pengujian Sentimen Real-Time:**
   - Input ulasan baru secara langsung dengan tombol ulasan cepat.
   - Penentuan label: **POSITIF 👍** vs **NEGATIF 👎**.
   - Menampilkan skor $\cos(r, c^+)$, $\cos(r, c^-)$, dan persentase keyakinan.
   - Menampilkan probabilitas model alternatif Regresi Logistik $P(\text{Pos})$ vs $P(\text{Neg})$.
   - Rincian tahapan NLP: Token bersih, deteksi kata kamus vs kata OOV.
   - Visualisasi interaktif grafik batang koordinat **Vektor 50 Dimensi (Vector Average)**.
   - Opsi simpan ulasan baru langsung ke basis data relasional SQLite.

3. **Simulasi Matematis GloVe (Proposal Bab 4):**
   - Tab 4.1: Matriks Kookurensi $X(i,j)$ dengan kustomisasi ukuran jendela & pembobotan jarak.
   - Tab 4.2: Langkah perhitungan SGD (Data, Bobot, Prediksi, Selisih, Loss, Gradien, Pembaruan Parameter, Evaluasi Penurunan Loss).
   - Tab 4.3: Perhitungan Cosine Similarity 2D dan **Peta Sebaran Visual Ruang Vektor 2D** antara kata, sentroid, dan ulasan baru.

4. **Import CSV & Basis Data:**
   - Formulir unggah berkas CSV ulasan produk baru dengan klasifikasi sentimen otomatis.
   - Penjelajah isi 6 tabel basis data relasional (`PENJUAL`, `PRODUK`, `ULASAN`, `MODEL_GLOVE`, `VEKTOR_KATA`, `KOOKURENSI`, `HASIL_SENTIMEN`).
   - Tombol ekspor dan unduh data tabel dalam format CSV.

5. **Perancangan Sistem:**
   - Dokumentasi lengkap DFD Level 0, DFD Level 1, dan ERD.

