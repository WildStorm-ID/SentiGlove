"""
Pydantic Schemas untuk Validasi Request & Response FastAPI SentiGlove
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# === Schema Prediksi Sentimen ===
class PredictRequest(BaseModel):
    text: str = Field(..., description="Teks ulasan e-commerce yang ingin diuji", min_length=1)
    save_to_db: bool = Field(False, description="Apakah ulasan dan prediksi disimpan ke database")
    product_id: Optional[str] = Field(None, description="ID Produk jika ingin dikaitkan")


class TokenInfo(BaseModel):
    total_tokens: int
    matched_count: int
    oov_count: int
    coverage_pct: float
    matched_words: List[str]
    oov_words: List[str]


class PredictResponse(BaseModel):
    original_text: str
    tokens: List[str]
    token_info: TokenInfo
    label: str
    skor_pos: float
    skor_neg: float
    confidence: float
    logistic_label: str
    logistic_prob_pos: float
    logistic_prob_neg: float
    vector: List[float]
    saved_review_id: Optional[int] = None


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    total: int
    results: List[PredictResponse]


# === Schema Simulasi Perhitungan GloVe ===
class CooccurrenceSimRequest(BaseModel):
    sentences: List[str] = Field(..., min_length=1)
    window_size: int = Field(1, ge=1, le=10)
    distance_weighting: bool = Field(False)


class CooccurrenceSimResponse(BaseModel):
    vocab: List[str]
    matrix: Dict[str, Dict[str, float]]


class GloveStepSimRequest(BaseModel):
    x_ij: float = Field(2.0, gt=0)
    w_i: List[float] = Field(default_factory=lambda: [0.30, 0.10])
    w_tilde_j: List[float] = Field(default_factory=lambda: [0.20, 0.40])
    b_i: float = Field(0.10)
    b_tilde_j: float = Field(0.05)
    x_max: float = Field(10.0, gt=0)
    alpha: float = Field(0.75, gt=0)
    eta: float = Field(0.10, gt=0)


class GloveStepSimResponse(BaseModel):
    x_ij: float
    log_x: float
    f_val: float
    dot_product: float
    pred: float
    d: float
    J: float
    g: float
    grad_w_i: List[float]
    grad_w_tilde_j: List[float]
    grad_b_i: float
    grad_b_tilde_j: float
    w_i_old: List[float]
    w_tilde_j_old: List[float]
    b_i_old: float
    b_tilde_j_old: float
    w_i_new: List[float]
    w_tilde_j_new: List[float]
    b_i_new: float
    b_tilde_j_new: float
    pred_new: float
    d_new: float
    J_new: float
    loss_reduction: float


# === Schema Data E-commerce & Produk ===
class ProductSummary(BaseModel):
    id_produk: str
    nama_produk: str
    kategori: str
    id_penjual: str
    nama_toko: str
    total_ulasan: int
    total_positif: int
    total_negatif: int
    rasio_positif_pct: float
    rata_rating: float


class ReviewItem(BaseModel):
    id_ulasan: int
    id_produk: str
    nama_produk: Optional[str] = None
    teks: str
    rating_asli: Optional[int]
    label_asli: Optional[str]
    label_prediksi: str
    skor_pos: float
    skor_neg: float
    created_at: str


class DashboardStats(BaseModel):
    total_penjual: int
    total_produk: int
    total_ulasan: int
    total_positif: int
    total_negatif: int
    persentase_positif: float
    rata_rating_keseluruhan: float

