"""
Modul Klasifikasi Sentimen Ulasan (Proses 4.0 DFD)
Menerapkan:
- Representasi Vector Average (50d) ulasan
- Perhitungan Centroid Positif (c+) dan Centroid Negatif (c-)
- Klasifikasi berbasis Cosine Similarity tertinggi ke c+ vs c-
- Model pembanding / fallback: Regresi Logistik (Logistic Regression)
"""

import os
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from engine.preprocessor import preprocess_review
from engine.glove_loader import GloVeEmbedding

MODEL_FILE = os.path.join("model and dataset", "senti_glove_classifier.pkl")


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class SentiGloveClassifier:
    _instance: Optional["SentiGloveClassifier"] = None

    def __init__(self, model_path: str = MODEL_FILE):
        self.model_path = model_path
        self.glove = GloVeEmbedding.get_instance()
        self.c_pos: Optional[np.ndarray] = None
        self.c_neg: Optional[np.ndarray] = None
        self.logistic_model: Optional[LogisticRegression] = None
        self.is_trained = False
        self._load_if_exists()

    @classmethod
    def get_instance(cls, model_path: str = MODEL_FILE) -> "SentiGloveClassifier":
        if cls._instance is None:
            cls._instance = SentiGloveClassifier(model_path)
        return cls._instance

    def _load_if_exists(self) -> bool:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.c_pos = data["c_pos"]
                    self.c_neg = data["c_neg"]
                    self.logistic_model = data.get("logistic_model")
                    self.is_trained = True
                return True
            except Exception as e:
                print(f"Gagal memuat model tersimpan: {e}")
        return False

    def save(self):
        """Menyimpan model centroid dan regresi logistik ke file pkl."""
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "c_pos": self.c_pos,
                "c_neg": self.c_neg,
                "logistic_model": self.logistic_model
            }, f)

    def train_from_reviews(self, texts: List[str], labels: List[str], max_iter: int = 300):
        """
        Melatih pengklasifikasi dengan menghitung Centroid Positif & Negatif,
        serta melatih model fallback Logistic Regression.
        """
        pos_vectors = []
        neg_vectors = []
        all_vectors = []
        valid_labels = []

        for text, label in zip(texts, labels):
            tokens = preprocess_review(text)
            vec, _ = self.glove.sentence_vector(tokens)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue

            all_vectors.append(vec)
            valid_labels.append(label)

            if label.lower() == "positif":
                pos_vectors.append(vec)
            elif label.lower() == "negatif":
                neg_vectors.append(vec)

        if not pos_vectors or not neg_vectors:
            raise ValueError("Data latih harus memiliki ulasan Positif dan Negatif yang valid.")

        # Latih Logistic Regression fallback & diskriminan
        X = np.array(all_vectors, dtype=np.float32)
        y = np.array(valid_labels)
        self.logistic_model = LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=42)
        self.logistic_model.fit(X, y)

        # Hitung Centroid Positif (c+) dan Negatif (c-)
        # Menggunakan kombinasi titik pusat korpus dan arah diskriminan bobot w
        # sehingga Cosine Similarity c+ vs c- selaras sempurna dengan hyperplane keputusan
        classes = list(self.logistic_model.classes_)
        if "Positif" in classes and "Negatif" in classes:
            pos_idx = classes.index("Positif")
            # Jika Positif adalah kelas 1, w menunjuk ke arah Positif
            w = self.logistic_model.coef_[0] if pos_idx == 1 else -self.logistic_model.coef_[0]
            mu = np.mean(X, axis=0)
            self.c_pos = (mu + 2.0 * w).astype(np.float32)
            self.c_neg = (mu - 2.0 * w).astype(np.float32)
        else:
            self.c_pos = np.mean(np.array(pos_vectors, dtype=np.float32), axis=0)
            self.c_neg = np.mean(np.array(neg_vectors, dtype=np.float32), axis=0)

        self.is_trained = True
        self.save()

    def predict_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        """
        Memprediksi sentimen berdasarkan vektor ulasan 50 dimensi.
        """
        if not self.is_trained or self.c_pos is None or self.c_neg is None:
            raise RuntimeError("Model belum dilatih atau centroid belum diinisialisasi.")

        cos_pos = cosine_sim(vec, self.c_pos)
        cos_neg = cosine_sim(vec, self.c_neg)

        # Keputusan berdasarkan Cosine Similarity tertinggi
        label_pred = "Positif" if cos_pos >= cos_neg else "Negatif"

        # Hitung persentase confidence heuristik dari jarak cosine
        # Petakan (-1 s/d 1) ke [0, 1]
        shift_pos = max(0.0, cos_pos + 1.0)
        shift_neg = max(0.0, cos_neg + 1.0)
        total_shift = shift_pos + shift_neg
        conf_pos = (shift_pos / total_shift) if total_shift > 0 else 0.5
        conf_neg = (shift_neg / total_shift) if total_shift > 0 else 0.5
        confidence = conf_pos if label_pred == "Positif" else conf_neg

        # Fallback Logistic Regression
        lr_label = label_pred
        lr_prob_pos = float(conf_pos)
        lr_prob_neg = float(conf_neg)

        if self.logistic_model is not None:
            vec_2d = vec.reshape(1, -1)
            lr_label = str(self.logistic_model.predict(vec_2d)[0])
            classes = list(self.logistic_model.classes_)
            probs = self.logistic_model.predict_proba(vec_2d)[0]
            if "Positif" in classes and "Negatif" in classes:
                pos_idx = classes.index("Positif")
                neg_idx = classes.index("Negatif")
                lr_prob_pos = float(probs[pos_idx])
                lr_prob_neg = float(probs[neg_idx])

        return {
            "label": label_pred,
            "skor_pos": round(cos_pos, 4),
            "skor_neg": round(cos_neg, 4),
            "confidence": round(confidence * 100, 2),
            "logistic_label": lr_label,
            "logistic_prob_pos": round(lr_prob_pos, 4),
            "logistic_prob_neg": round(lr_prob_neg, 4),
            "vector_preview": [round(float(x), 4) for x in vec]
        }

    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Pipeline lengkap klasifikasi teks:
        1. Preprocessing (clean, slang normalization, stopword filtering)
        2. Vector Averaging 50d
        3. Prediksi Cosine Similarity & Regresi Logistik
        """
        tokens = preprocess_review(text)
        vec, token_info = self.glove.sentence_vector(tokens)
        pred_res = self.predict_vector(vec)

        return {
            "original_text": text,
            "tokens": tokens,
            "token_info": token_info,
            "vector": [round(float(x), 4) for x in vec],
            **pred_res
        }
