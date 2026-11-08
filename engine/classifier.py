"""
Modul Klasifikasi Sentimen Ulasan (Proses 4.0 DFD)
Menerapkan:
- Representasi Vector Average (50d) ulasan
- Perhitungan Centroid Positif (c+) dan Centroid Negatif (c-)
- Klasifikasi berbasis Cosine Similarity tertinggi ke c+ vs c-
- Model pembanding / fallback: Regresi Logistik (Logistic Regression)
- Fitur Self-Healing: Otomatis melatih dan menyimpan model jika file pkl belum ada
"""

import os
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from engine.preprocessor import preprocess_review
from engine.glove_loader import GloVeEmbedding

MODEL_FILE = os.path.join("model and dataset", "senti_glove_classifier.pkl")
CSV_PATH = os.path.join("model and dataset", "tokopedia-product-reviews-2019.csv")


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class SentiGloveClassifier:
    _instance: Optional["SentiGloveClassifier"] = None

    def __init__(self, model_path: str = MODEL_FILE, csv_path: str = CSV_PATH):
        self.model_path = model_path
        self.csv_path = csv_path
        self.glove = GloVeEmbedding.get_instance()
        self.c_pos: Optional[np.ndarray] = None
        self.c_neg: Optional[np.ndarray] = None
        self.logistic_model: Optional[LogisticRegression] = None
        self.is_trained = False
        
        # Coba muat model tersimpan; jika belum ada, inisialisasi / latih otomatis
        if not self._load_if_exists():
            self._auto_init()

    @classmethod
    def get_instance(cls, model_path: str = MODEL_FILE, csv_path: str = CSV_PATH) -> "SentiGloveClassifier":
        if cls._instance is None:
            cls._instance = SentiGloveClassifier(model_path, csv_path)
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
                print(f"[Warning] Gagal memuat model tersimpan ({self.model_path}): {e}")
        return False

    def _auto_init(self):
        """Self-healing: Melatih model dari dataset Tokopedia atau seed kata jika model belum ada."""
        if os.path.exists(self.csv_path):
            try:
                print(">>> [Auto-Init] Melatih SentiGlove Classifier dari dataset ulasan Tokopedia...")
                df = pd.read_csv(self.csv_path)
                df = df[df["rating"] != 3].copy()
                df["label"] = df["rating"].apply(lambda r: "Positif" if r >= 4 else "Negatif")

                df_neg = df[df["label"] == "Negatif"]
                df_pos = df[df["label"] == "Positif"].sample(n=min(len(df_neg) * 2, 2000), random_state=42)
                df_train = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42)

                train_texts = df_train["text"].dropna().tolist()
                train_labels = df_train["label"].tolist()
                self.train_from_reviews(train_texts, train_labels)
                print(">>> [Auto-Init] SentiGlove Classifier berhasil dilatih dan disimpan otomatis!")
                return
            except Exception as e:
                print(f"[Warning] Gagal melatih otomatis dari dataset: {e}")

        # Fallback: Inisialisasi centroid dari kata sentimen representatif (Proposal Bab 4.3)
        print(">>> [Auto-Init] Menggunakan fallback sentimen berbasis kata seed representatif...")
        pos_seeds = ["bagus", "mantap", "puas", "cepat", "rapi", "asli", "murah", "rekomendasi", "keren", "terbaik"]
        neg_seeds = ["jelek", "rusak", "kecewa", "lambat", "pecah", "cacat", "parah", "hancur", "palsu", "kapok"]
        
        pos_vecs = [self.glove.get_word_vector(w)[0] for w in pos_seeds if self.glove.get_word_vector(w)[1]]
        neg_vecs = [self.glove.get_word_vector(w)[0] for w in neg_seeds if self.glove.get_word_vector(w)[1]]

        self.c_pos = np.mean(pos_vecs, axis=0).astype(np.float32)
        self.c_neg = np.mean(neg_vecs, axis=0).astype(np.float32)
        self.is_trained = True
        self.save()

    def save(self):
        """Menyimpan model centroid dan regresi logistik ke file pkl."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "c_pos": self.c_pos,
                    "c_neg": self.c_neg,
                    "logistic_model": self.logistic_model
                }, f)
        except Exception as e:
            print(f"[Warning] Gagal menyimpan model ke {self.model_path}: {e}")

    def train_from_reviews(self, texts: List[str], labels: List[str], max_iter: int = 400):
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
        # sehingga Cosine Similarity c+ vs c- selaras dengan hyperplane keputusan
        classes = list(self.logistic_model.classes_)
        if "Positif" in classes and "Negatif" in classes:
            pos_idx = classes.index("Positif")
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
