"""
Modul Simulasi Perhitungan GloVe (Proses 3.0 DFD & Bab 4 Proposal UTS)
Menyediakan kalkulasi matematis step-by-step:
- 4.1 Matriks Kookurensi
- 4.2 Fungsi Bobot, Loss J, Gradien, dan Pembaruan SGD (1 Iterasi)
- 4.3 Klasifikasi Sentimen Ulasan Baru dengan Cosine Similarity ke Centroid (c+ vs c-)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from engine.cooccurrence import build_cooccurrence_matrix


def glove_weight(x: float, x_max: float = 10.0, alpha: float = 0.75) -> float:
    """Fungsi pembobot GloVe: f(x) = (x / x_max)^alpha jika x < x_max else 1.0."""
    if x <= 0:
        return 0.0
    if x < x_max:
        return float((x / x_max) ** alpha)
    return 1.0


def glove_step_calculation(
    x_ij: float,
    w_i: np.ndarray,
    w_tilde_j: np.ndarray,
    b_i: float,
    b_tilde_j: float,
    x_max: float = 10.0,
    alpha: float = 0.75,
    eta: float = 0.1
) -> Dict[str, Any]:
    """
    Menghitung 1 langkah pembaruan SGD GloVe persis sesuai Tabel 4 Proposal UTS.
    
    Langkah:
    1. Data: Xij, log(Xij)
    2. Bobot: f(Xij) = (Xij / x_max)^alpha
    3. Prediksi: dot(w_i, w_tilde_j) + b_i + b_tilde_j
    4. Selisih: d = prediksi - log(Xij)
    5. Loss: J = f * d^2
    6. Gradien: g = 2 * f * d
       grad_w_i = g * w_tilde_j
       grad_w_tilde_j = g * w_i
       grad_b_i = g
       grad_b_tilde_j = g
    7. Pembaruan:
       w_i_new = w_i - eta * grad_w_i
       w_tilde_j_new = w_tilde_j - eta * grad_w_tilde_j
       b_i_new = b_i - eta * grad_b_i
       b_tilde_j_new = b_tilde_j - eta * grad_b_tilde_j
    8. Cek ulang: pred_new, d_new, J_new
    """
    w_i = np.array(w_i, dtype=np.float64)
    w_tilde_j = np.array(w_tilde_j, dtype=np.float64)

    log_x = float(np.log(x_ij))
    f_val = glove_weight(x_ij, x_max, alpha)

    dot_prod = float(np.dot(w_i, w_tilde_j))
    pred = dot_prod + float(b_i) + float(b_tilde_j)
    d = pred - log_x
    J = f_val * (d ** 2)

    g = 2.0 * f_val * d
    grad_w_i = g * w_tilde_j
    grad_w_tilde_j = g * w_i
    grad_b_i = g
    grad_b_tilde_j = g

    w_i_new = w_i - eta * grad_w_i
    w_tilde_j_new = w_tilde_j - eta * grad_w_tilde_j
    b_i_new = float(b_i - eta * grad_b_i)
    b_tilde_j_new = float(b_tilde_j - eta * grad_b_tilde_j)

    pred_new = float(np.dot(w_i_new, w_tilde_j_new) + b_i_new + b_tilde_j_new)
    d_new = pred_new - log_x
    J_new = f_val * (d_new ** 2)

    return {
        "x_ij": round(x_ij, 4),
        "log_x": round(log_x, 4),
        "f_val": round(f_val, 4),
        "dot_product": round(dot_prod, 4),
        "pred": round(pred, 4),
        "d": round(d, 4),
        "J": round(J, 6),
        "g": round(g, 4),
        "grad_w_i": np.round(grad_w_i, 4).tolist(),
        "grad_w_tilde_j": np.round(grad_w_tilde_j, 4).tolist(),
        "grad_b_i": round(grad_b_i, 4),
        "grad_b_tilde_j": round(grad_b_tilde_j, 4),
        "w_i_old": np.round(w_i, 4).tolist(),
        "w_tilde_j_old": np.round(w_tilde_j, 4).tolist(),
        "b_i_old": round(float(b_i), 4),
        "b_tilde_j_old": round(float(b_tilde_j), 4),
        "w_i_new": np.round(w_i_new, 4).tolist(),
        "w_tilde_j_new": np.round(w_tilde_j_new, 4).tolist(),
        "b_i_new": round(b_i_new, 4),
        "b_tilde_j_new": round(b_tilde_j_new, 4),
        "pred_new": round(pred_new, 4),
        "d_new": round(d_new, 4),
        "J_new": round(J_new, 6),
        "loss_reduction": round(J - J_new, 6)
    }


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Menghitung Cosine Similarity antara dua vektor."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def get_proposal_simulation_data() -> Dict[str, Any]:
    """
    Mengembalikan data simulasi lengkap persis sesuai Bab 4 Proposal:
    - 4.1 Korpus 3 ulasan, window 1, tabel matriks kookurensi
    - 4.2 Simulasi bobot, loss, SGD untuk pasangan barang-bagus
    - 4.3 Vektor kata 2D, centroid positif/negatif, dan prediksi 2 ulasan baru
    """
    # 4.1 Korpus Tiga Ulasan
    raw_corpus = [
        "barang bagus sekali",
        "barang jelek sekali",
        "pengiriman cepat barang bagus"
    ]
    tokenized_corpus = [s.split() for s in raw_corpus]
    vocab_order = ["barang", "bagus", "jelek", "sekali", "cepat", "pengiriman"]
    vocab, X, df_X = build_cooccurrence_matrix(
        tokenized_corpus, vocab=vocab_order, window_size=1, distance_weighting=False
    )

    # 4.2 Simulasi Pasangan "barang" - "bagus"
    w_barang = np.array([0.30, 0.10])
    w_tilde_bagus = np.array([0.20, 0.40])
    b_barang = 0.10
    b_tilde_bagus = 0.05
    step_res = glove_step_calculation(
        x_ij=2.0,
        w_i=w_barang,
        w_tilde_j=w_tilde_bagus,
        b_i=b_barang,
        b_tilde_j=b_tilde_bagus,
        x_max=10.0,
        alpha=0.75,
        eta=0.1
    )

    # 4.3 Vektor Kata 2D & Klasifikasi Sentimen
    word_vectors_2d = {
        "barang": np.array([0.30, 0.10]),
        "bagus": np.array([0.60, 0.50]),
        "cepat": np.array([0.45, 0.30]),
        "jelek": np.array([-0.55, -0.45]),
        "sekali": np.array([0.10, 0.05])
    }
    # Pusat kelas
    c_pos = (word_vectors_2d["bagus"] + word_vectors_2d["cepat"]) / 2.0  # [0.525, 0.400]
    c_neg = word_vectors_2d["jelek"].copy()                             # [-0.550, -0.450]

    # Ulasan baru
    new_reviews = [
        {"text": "barang cepat bagus", "tokens": ["barang", "cepat", "bagus"]},
        {"text": "barang jelek sekali", "tokens": ["barang", "jelek", "sekali"]}
    ]

    review_results = []
    for r in new_reviews:
        vecs = [word_vectors_2d[w] for w in r["tokens"]]
        r_vec = np.mean(vecs, axis=0)
        cos_pos = cosine_similarity(r_vec, c_pos)
        cos_neg = cosine_similarity(r_vec, c_neg)
        pred_label = "Positif" if cos_pos >= cos_neg else "Negatif"
        review_results.append({
            "text": r["text"],
            "r_vec": np.round(r_vec, 3).tolist(),
            "cos_pos": round(cos_pos, 3),
            "cos_neg": round(cos_neg, 3),
            "pred": pred_label
        })

    return {
        "corpus": raw_corpus,
        "vocab": vocab,
        "cooccurrence_matrix": df_X.to_dict(),
        "sgd_step": step_res,
        "word_vectors_2d": {k: v.tolist() for k, v in word_vectors_2d.items()},
        "c_pos": np.round(c_pos, 3).tolist(),
        "c_neg": np.round(c_neg, 3).tolist(),
        "review_results": review_results
    }

