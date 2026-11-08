"""
Router API FastAPI untuk Layanan SentiGlove
"""

import io
from typing import List, Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.db import get_db
from database.models import Penjual, Produk, Ulasan, ModelGlove, HasilSentimen, get_utc_now
from engine.classifier import SentiGloveClassifier
from engine.cooccurrence import build_cooccurrence_matrix
from engine.simulation import glove_step_calculation, get_proposal_simulation_data
from api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    CooccurrenceSimRequest, CooccurrenceSimResponse,
    GloveStepSimRequest, GloveStepSimResponse,
    ProductSummary, ReviewItem, DashboardStats
)

router = APIRouter(prefix="/api", tags=["SentiGlove"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Memeriksa kesehatan sistem, ketersediaan model, dan jumlah data di database."""
    classifier = SentiGloveClassifier.get_instance()
    total_ulasan = db.query(Ulasan).count()
    total_produk = db.query(Produk).count()
    return {
        "status": "healthy",
        "model_trained": classifier.is_trained,
        "glove_dimension": classifier.glove.dimension,
        "total_ulasan_db": total_ulasan,
        "total_produk_db": total_produk
    }


@router.post("/predict", response_model=PredictResponse)
def predict_sentiment(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Mengklasifikasikan sentimen sebuah teks ulasan:
    - Preprocessing
    - Vector Averaging 50d
    - Centroid Cosine Similarity & Regresi Logistik
    - Opsi menyimpan ke basis data
    """
    classifier = SentiGloveClassifier.get_instance()
    res = classifier.predict_text(req.text)

    saved_id = None
    if req.save_to_db:
        # Cari produk terkait atau default
        prod_id = req.product_id
        if not prod_id:
            first_prod = db.query(Produk).first()
            prod_id = first_prod.id_produk if first_prod else "prod_live"

        ulasan_obj = Ulasan(
            id_produk=prod_id,
            teks=req.text,
            rating_asli=5 if res["label"] == "Positif" else 1,
            label_asli=res["label"],
            created_at=get_utc_now()
        )
        db.add(ulasan_obj)
        db.flush()

        hasil_obj = HasilSentimen(
            id_ulasan=ulasan_obj.id_ulasan,
            id_model="glove_wiki_id_50d",
            skor_pos=res["skor_pos"],
            skor_neg=res["skor_neg"],
            label_prediksi=res["label"],
            created_at=get_utc_now()
        )
        db.add(hasil_obj)
        db.commit()
        saved_id = ulasan_obj.id_ulasan

    return {
        **res,
        "saved_review_id": saved_id
    }


@router.post("/predict/batch", response_model=BatchPredictResponse)
def batch_predict_sentiment(req: BatchPredictRequest):
    """Mengklasifikasikan sekumpulan teks ulasan secara batch."""
    classifier = SentiGloveClassifier.get_instance()
    results = [classifier.predict_text(t) for t in req.texts]
    return {
        "total": len(results),
        "results": results
    }


@router.post("/simulate/cooccurrence", response_model=CooccurrenceSimResponse)
def simulate_cooccurrence(req: CooccurrenceSimRequest):
    """
    Menghitung matriks kookurensi X dari kalimat input
    dengan window size dan pembobotan jarak kustom.
    """
    tokenized = [s.strip().split() for s in req.sentences if s.strip()]
    if not tokenized:
        raise HTTPException(status_code=400, detail="Kalimat input tidak boleh kosong.")
    vocab, _, df_matrix = build_cooccurrence_matrix(
        tokenized, window_size=req.window_size, distance_weighting=req.distance_weighting
    )
    return {
        "vocab": vocab,
        "matrix": df_matrix.to_dict()
    }


@router.post("/simulate/glove-step", response_model=GloveStepSimResponse)
def simulate_glove_step(req: GloveStepSimRequest):
    """Menghitung 1 iterasi pembaruan SGD GloVe matematis sesuai Bab 4.2 Proposal."""
    res = glove_step_calculation(
        x_ij=req.x_ij,
        w_i=req.w_i,
        w_tilde_j=req.w_tilde_j,
        b_i=req.b_i,
        b_tilde_j=req.b_tilde_j,
        x_max=req.x_max,
        alpha=req.alpha,
        eta=req.eta
    )
    return res


@router.get("/simulate/proposal")
def simulate_proposal_full():
    """Mengembalikan data simulasi proposal Bab 4 secara lengkap (Tabel 3, Tabel 4, Tabel 5)."""
    return get_proposal_simulation_data()


@router.get("/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Mengembalikan statistik ringkasan toko dan proporsi sentimen untuk dashboard penjual."""
    total_penjual = db.query(Penjual).count()
    total_produk = db.query(Produk).count()
    total_ulasan = db.query(Ulasan).count()

    total_pos = db.query(HasilSentimen).filter_by(label_prediksi="Positif").count()
    total_neg = db.query(HasilSentimen).filter_by(label_prediksi="Negatif").count()

    pct_pos = round((total_pos / total_ulasan * 100), 2) if total_ulasan > 0 else 0.0

    avg_rating_res = db.query(func.avg(Ulasan.rating_asli)).scalar()
    avg_rating = round(float(avg_rating_res), 2) if avg_rating_res else 0.0

    return {
        "total_penjual": total_penjual,
        "total_produk": total_produk,
        "total_ulasan": total_ulasan,
        "total_positif": total_pos,
        "total_negatif": total_neg,
        "persentase_positif": pct_pos,
        "rata_rating_keseluruhan": avg_rating
    }


@router.get("/products", response_model=List[ProductSummary])
def list_products(
    kategori: Optional[str] = None,
    id_penjual: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Mendapatkan daftar produk beserta agregasi sentimen dan rating ulasannya."""
    query = db.query(Produk)
    if kategori:
        query = query.filter(Produk.kategori == kategori)
    if id_penjual:
        query = query.filter(Produk.id_penjual == id_penjual)

    products = query.limit(limit).all()
    results = []

    for p in products:
        ulasan_list = p.ulasans
        n_ulasan = len(ulasan_list)
        n_pos = sum(1 for u in ulasan_list if u.hasil_sentimen and u.hasil_sentimen.label_prediksi == "Positif")
        n_neg = sum(1 for u in ulasan_list if u.hasil_sentimen and u.hasil_sentimen.label_prediksi == "Negatif")
        pct_pos = round((n_pos / n_ulasan * 100), 2) if n_ulasan > 0 else 0.0

        ratings = [u.rating_asli for u in ulasan_list if u.rating_asli is not None]
        avg_rat = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

        results.append(ProductSummary(
            id_produk=p.id_produk,
            nama_produk=p.nama_produk,
            kategori=p.kategori,
            id_penjual=p.id_penjual,
            nama_toko=p.penjual.nama_toko if p.penjual else p.id_penjual,
            total_ulasan=n_ulasan,
            total_positif=n_pos,
            total_negatif=n_neg,
            rasio_positif_pct=pct_pos,
            rata_rating=avg_rat
        ))

    return results


@router.get("/reviews", response_model=List[ReviewItem])
def list_reviews(
    id_produk: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Mendapatkan daftar ulasan dengan opsi filter produk dan label sentimen."""
    query = db.query(Ulasan).join(HasilSentimen, Ulasan.id_ulasan == HasilSentimen.id_ulasan)

    if id_produk:
        query = query.filter(Ulasan.id_produk == id_produk)
    if sentiment:
        query = query.filter(HasilSentimen.label_prediksi == sentiment)

    ulasan_records = query.order_by(Ulasan.id_ulasan.desc()).offset(offset).limit(limit).all()

    items = []
    for u in ulasan_records:
        hs = u.hasil_sentimen
        items.append(ReviewItem(
            id_ulasan=u.id_ulasan,
            id_produk=u.id_produk,
            nama_produk=u.produk.nama_produk if u.produk else None,
            teks=u.teks,
            rating_asli=u.rating_asli,
            label_asli=u.label_asli,
            label_prediksi=hs.label_prediksi if hs else "Tidak Diketahui",
            skor_pos=hs.skor_pos if hs else 0.0,
            skor_neg=hs.skor_neg if hs else 0.0,
            created_at=u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else ""
        ))
    return items


@router.post("/import-csv")
async def import_csv_reviews(
    file: UploadFile = File(...),
    max_rows: int = Query(200, ge=1, le=2000),
    db: Session = Depends(get_db)
):
    """
    Mengimpor file CSV ulasan produk, mengklasifikasikan secara otomatis dengan SentiGlove,
    dan menyimpannya ke database relasional.
    """
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca format CSV: {e}")

    # Kolom wajib minimal ada 'text'
    text_col = None
    for candidate in ["text", "ulasan", "review", "komentar"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if not text_col:
        raise HTTPException(status_code=400, detail="CSV harus memiliki kolom teks ulasan ('text' atau 'ulasan').")

    # Kolom rating jika ada
    rating_col = None
    for candidate in ["rating", "bintang", "score"]:
        if candidate in df.columns:
            rating_col = candidate
            break

    prod_name_col = "product_name" if "product_name" in df.columns else None
    prod_id_col = "product_id" if "product_id" in df.columns else None
    cat_col = "category" if "category" in df.columns else None
    shop_id_col = "shop_id" if "shop_id" in df.columns else None

    # Filter rating 3 jika ada rating
    if rating_col:
        df = df[df[rating_col] != 3]

    df_subset = df.head(max_rows)
    classifier = SentiGloveClassifier.get_instance()

    penjual_cache = {p[0] for p in db.query(Penjual.id_penjual).all()}
    produk_cache = {p[0] for p in db.query(Produk.id_produk).all()}

    imported_count = 0
    pos_count = 0
    neg_count = 0

    for idx, row in df_subset.iterrows():
        review_text = str(row[text_col]).strip()
        if not review_text or review_text.lower() == "nan":
            continue

        rating_val = int(row[rating_col]) if rating_col and pd.notnull(row[rating_col]) else None
        label_asli = ("Positif" if rating_val >= 4 else "Negatif") if rating_val is not None else None

        shop_id = str(row[shop_id_col]) if shop_id_col and pd.notnull(row[shop_id_col]) else "toko_import"
        prod_id = str(row[prod_id_col]) if prod_id_col and pd.notnull(row[prod_id_col]) else f"prod_import_{idx}"
        prod_name = str(row[prod_name_col]) if prod_name_col and pd.notnull(row[prod_name_col]) else f"Produk Import {idx}"
        category = str(row[cat_col]) if cat_col and pd.notnull(row[cat_col]) else "Umum"

        if shop_id not in penjual_cache:
            db.add(Penjual(id_penjual=shop_id, nama_toko=f"Toko #{shop_id}"))
            db.flush()
            penjual_cache.add(shop_id)

        if prod_id not in produk_cache:
            db.add(Produk(id_produk=prod_id, id_penjual=shop_id, nama_produk=prod_name, kategori=category))
            db.flush()
            produk_cache.add(prod_id)

        ulasan_obj = Ulasan(
            id_produk=prod_id,
            teks=review_text,
            rating_asli=rating_val,
            label_asli=label_asli,
            created_at=get_utc_now()
        )
        db.add(ulasan_obj)
        db.flush()

        pred_res = classifier.predict_text(review_text)
        if pred_res["label"] == "Positif":
            pos_count += 1
        else:
            neg_count += 1

        db.add(HasilSentimen(
            id_ulasan=ulasan_obj.id_ulasan,
            id_model="glove_wiki_id_50d",
            skor_pos=pred_res["skor_pos"],
            skor_neg=pred_res["skor_neg"],
            label_prediksi=pred_res["label"],
            created_at=get_utc_now()
        ))
        imported_count += 1

    db.commit()

    return {
        "message": f"Berhasil mengimpor dan mengklasifikasikan {imported_count} ulasan.",
        "imported_count": imported_count,
        "positif": pos_count,
        "negatif": neg_count
    }

