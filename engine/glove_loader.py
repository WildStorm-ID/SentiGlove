"""
Modul Pemuatan dan Representasi Word Embedding GloVe 50 Dimensi.
Mendukung:
- Pemuatan cepat via file binary cache .npz (1-2 detik)
- Pencarian vektor kata (50d) dengan OOV fallback ke vektor nol 50d
- Rata-rata vektor kata (Vector Averaging) untuk representasi ulasan u = (1/N) * sum(w_i)
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np

DEFAULT_NPZ_PATH = os.path.join("model and dataset", "glove_50dim.npz")
DEFAULT_TXT_PATH = os.path.join("model and dataset", "glove_50dim_wiki.id.case.text.txt")
DIMENSION = 50


class GloVeEmbedding:
    _instance: Optional["GloVeEmbedding"] = None

    def __init__(self, npz_path: str = DEFAULT_NPZ_PATH, txt_path: str = DEFAULT_TXT_PATH):
        self.npz_path = npz_path
        self.txt_path = txt_path
        self.dimension = DIMENSION
        self.embeddings: Dict[str, np.ndarray] = {}
        self.is_loaded = False
        self._load()

    @classmethod
    def get_instance(cls, npz_path: str = DEFAULT_NPZ_PATH, txt_path: str = DEFAULT_TXT_PATH) -> "GloVeEmbedding":
        if cls._instance is None:
            cls._instance = GloVeEmbedding(npz_path, txt_path)
        return cls._instance

    def _load(self):
        """Memuat embedding dari .npz cache jika ada; jika tidak, dari .txt lalu menyimpannya ke .npz."""
        if os.path.exists(self.npz_path):
            data = np.load(self.npz_path)
            words = data["words"]
            vectors = data["vectors"]
            self.embeddings = {w: vectors[i] for i, w in enumerate(words)}
            self.is_loaded = True
            return

        if os.path.exists(self.txt_path):
            words = []
            vectors = []
            with open(self.txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == self.dimension + 1:
                        words.append(parts[0])
                        vectors.append([float(x) for x in parts[1:]])
            words_arr = np.array(words)
            vecs_arr = np.array(vectors, dtype=np.float32)
            # Simpan cache .npz agar ke depannya jauh lebih cepat
            np.savez_compressed(self.npz_path, words=words_arr, vectors=vecs_arr)
            self.embeddings = {w: vecs_arr[i] for i, w in enumerate(words_arr)}
            self.is_loaded = True
            return

        raise FileNotFoundError(
            f"Tidak menemukan file embedding di {self.npz_path} maupun {self.txt_path}"
        )

    def get_word_vector(self, word: str) -> Tuple[np.ndarray, bool]:
        """
        Mengembalikan vektor 50 dimensi untuk sebuah kata.
        Jika kata tidak ditemukan (OOV), mengembalikan vektor nol 50d dan status False.
        """
        if word in self.embeddings:
            return self.embeddings[word], True
        return np.zeros(self.dimension, dtype=np.float32), False

    def sentence_vector(self, tokens: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Menghitung Vector Averaging ulasan:
        r = (1 / |V|) * sum(w_i)
        Kata OOV bernilai vektor nol 50d.
        
        Mengembalikan:
        - vektor ulasan (np.ndarray 50d)
        - metadata rincian kata (kata cocok, kata OOV, proporsi cakupan)
        """
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32), {
                "total_tokens": 0,
                "matched_count": 0,
                "oov_count": 0,
                "coverage_pct": 0.0,
                "matched_words": [],
                "oov_words": []
            }

        matched_words = []
        oov_words = []
        vectors = []

        for token in tokens:
            vec, found = self.get_word_vector(token)
            vectors.append(vec)
            if found:
                matched_words.append(token)
            else:
                oov_words.append(token)

        # Rata-rata elemen per elemen (Vector Average)
        avg_vec = np.mean(np.array(vectors, dtype=np.float32), axis=0)

        info = {
            "total_tokens": len(tokens),
            "matched_count": len(matched_words),
            "oov_count": len(oov_words),
            "coverage_pct": round((len(matched_words) / len(tokens)) * 100, 2),
            "matched_words": matched_words,
            "oov_words": oov_words
        }

        return avg_vec, info

