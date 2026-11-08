"""
Skrip Inisialisasi dan Seeding Basis Data SentiGlove
- Membuat seluruh tabel ERD relasional
- Menyimpan metadata Model GloVe (50d Pretrained & 2d Proposal Simulation)
- Menyimpan Vektor Kata dan Kookurensi simulasi Bab 4
- Melatih Centroid Positif & Negatif serta Regresi Logistik
- Mengimpor sampel ulasan riil dari dataset Tokopedia ke dalam tabel PENJUAL, PRODUK, ULASAN, dan HASIL_SENTIMEN
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from database.db import Base, engine, SessionLocal
from database.models import Penjual, Produk, Ulasan, ModelGlove, VektorKata, Kookurensi, HasilSentimen, get_utc_now
from engine.preprocessor import preprocess_review
from engine.classifier import SentiGloveClassifier
from engine.simulation import get_proposal_simulation_data


CSV_PATH = os.path.join("model and dataset", "tokopedia-product-reviews-2019.csv")


def init_database(seed_sample_count: int = 600):
    print(">>> 1. Membuat tabel basis data relasional (SQLite)...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Pastikan Model Glove tercatat di database
        model_50d = db.query(ModelGlove).filter_by(id_model="glove_wiki_id_50d").first()
        if not model_50d:
            model_50d = ModelGlove(
                id_model="glove_wiki_id_50d",
                dimensi=50,
                x_max=100.0,
                alpha=0.75,
                loss_akhir=0.0125,
                tanggal_latih=get_utc_now(),
                keterangan="Pretrained GloVe 50-Dimensi Korpus Wikipedia Bahasa Indonesia"
            )
            db.add(model_50d)

        model_2d = db.query(ModelGlove).filter_by(id_model="glove_toy_proposal_2d").first()
        if not model_2d:
            model_2d = ModelGlove(
                id_model="glove_toy_proposal_2d",
                dimensi=2,
                x_max=10.0,
                alpha=0.75,
                loss_akhir=0.0437,
                tanggal_latih=get_utc_now(),
                keterangan="Model Simulasi Manual 2D Bab 4 Proposal UTS NLP"
            )
            db.add(model_2d)
        db.commit()

        # 2. Masukkan data simulasi Proposal (Tabel 3 & 4) ke VektorKata dan Kookurensi
        sim_data = get_proposal_simulation_data()
        if db.query(VektorKata).filter_by(id_model="glove_toy_proposal_2d").count() == 0:
            print(">>> 2. Menyimpan Vektor Kata simulasi Proposal Bab 4...")
            for w, vec in sim_data["word_vectors_2d"].items():
                bias_val = 0.10 if w == "barang" else (0.05 if w == "bagus" else 0.0)
                vk = VektorKata(
                    id_model="glove_toy_proposal_2d",
                    kata=w,
                    vektor_w=json.dumps(vec),
                    bias_b=bias_val
                )
                db.add(vk)
            db.commit()

        if db.query(Kookurensi).filter_by(id_model="glove_toy_proposal_2d").count() == 0:
            print(">>> 3. Menyimpan data Matriks Kookurensi simulasi Proposal...")
            matrix_dict = sim_data["cooccurrence_matrix"]
            for col_word, row_vals in matrix_dict.items():
                for row_word, count in row_vals.items():
                    if count > 0:
                        db.add(Kookurensi(
                            id_model="glove_toy_proposal_2d",
                            kata_i=row_word,
                            kata_j=col_word,
                            jumlah=float(count)
                        ))
            db.commit()

        # 3. Latih / Pastikan Classifier SentiGlove terlatih
        classifier = SentiGloveClassifier.get_instance()
        if not classifier.is_trained:
            print(">>> 4. Melatih Centroid Positif (c+), Centroid Negatif (c-), dan Regresi Logistik...")
            df = pd.read_csv(CSV_PATH)
            # Filter rating 3
            df = df[df["rating"] != 3].copy()
            df["label"] = df["rating"].apply(lambda r: "Positif" if r >= 4 else "Negatif")

            # Ambil sampel seimbang agar representasi centroid optimal
            df_neg = df[df["label"] == "Negatif"]
            df_pos = df[df["label"] == "Positif"].sample(n=min(len(df_neg) * 2, 2000), random_state=42)
            df_train = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42)

            train_texts = df_train["text"].dropna().tolist()
            train_labels = df_train["label"].tolist()
            classifier.train_from_reviews(train_texts, train_labels)
            print("     Pelatihan centroid & logistik berhasil disimpan!")

        # 4. Seeding data ulasan e-commerce ke Database
        existing_reviews = db.query(Ulasan).count()
        if existing_reviews == 0 and os.path.exists(CSV_PATH):
            print(f">>> 5. Mengimpor {seed_sample_count} sampel ulasan dari Tokopedia ke SQLite...")
            df_all = pd.read_csv(CSV_PATH)
            # Filter biner (buang rating 3 sesuai todo.md)
            df_clean = df_all[df_all["rating"] != 3].copy()
            df_clean["label"] = df_clean["rating"].apply(lambda r: "Positif" if r >= 4 else "Negatif")

            # Ambil sampel representatif (ada positif dan negatif dari berbagai kategori dan toko)
            sample_neg = df_clean[df_clean["label"] == "Negatif"].head(seed_sample_count // 3)
            sample_pos = df_clean[df_clean["label"] == "Positif"].head(seed_sample_count - len(sample_neg))
            df_seed = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=42)

            penjual_cache = set(db.query(Penjual.id_penjual).all())
            penjual_cache = {p[0] for p in penjual_cache}
            produk_cache = set(db.query(Produk.id_produk).all())
            produk_cache = {p[0] for p in produk_cache}

            for _, row in df_seed.iterrows():
                shop_id = str(row.get("shop_id", "toko_default"))
                nama_toko = f"Official Store #{shop_id}"
                prod_id = str(row.get("product_id", "prod_default"))
                prod_name = str(row.get("product_name", "Produk E-commerce"))
                category = str(row.get("category", "Umum"))
                review_text = str(row.get("text", "")).strip()
                rating_asli = int(row.get("rating", 5))
                label_asli = "Positif" if rating_asli >= 4 else "Negatif"

                if not review_text:
                    continue

                # Tambahkan Penjual jika belum ada
                if shop_id not in penjual_cache:
                    db.add(Penjual(id_penjual=shop_id, nama_toko=nama_toko))
                    db.flush()
                    penjual_cache.add(shop_id)

                # Tambahkan Produk jika belum ada
                if prod_id not in produk_cache:
                    db.add(Produk(
                        id_produk=prod_id,
                        id_penjual=shop_id,
                        nama_produk=prod_name,
                        kategori=category
                    ))
                    db.flush()
                    produk_cache.add(prod_id)

                # Tambahkan Ulasan
                ulasan_obj = Ulasan(
                    id_produk=prod_id,
                    teks=review_text,
                    rating_asli=rating_asli,
                    label_asli=label_asli,
                    created_at=get_utc_now()
                )
                db.add(ulasan_obj)
                db.flush()

                # Klasifikasikan sentimen ulasan secara otomatis menggunakan SentiGlove
                pred_result = classifier.predict_text(review_text)

                hasil_obj = HasilSentimen(
                    id_ulasan=ulasan_obj.id_ulasan,
                    id_model="glove_wiki_id_50d",
                    skor_pos=pred_result["skor_pos"],
                    skor_neg=pred_result["skor_neg"],
                    label_prediksi=pred_result["label"],
                    created_at=get_utc_now()
                )
                db.add(hasil_obj)

            db.commit()
            print(f"     Berhasil menyimpan ulasan dan hasil sentimen ke database!")
        else:
            print(f">>> Database sudah berisi {existing_reviews} ulasan.")

        print(">>> Inisialisasi database SentiGlove selesai dengan sukses!\n")

    except Exception as e:
        db.rollback()
        print(f"Error inisialisasi database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_database()
