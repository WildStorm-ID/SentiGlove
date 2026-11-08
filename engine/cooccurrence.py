"""
Modul Perhitungan Matriks Kookurensi X (Proses 2.0 DFD)
Mendukung:
- Jendela konteks simetris (window size)
- Pembobotan jarak (weight = 1 / jarak atau flat 1.0)
- Ekstraksi korpus ke matriks kookurensi X(i, j)
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd


def build_cooccurrence_matrix(
    tokenized_corpus: List[List[str]],
    vocab: Optional[List[str]] = None,
    window_size: int = 1,
    distance_weighting: bool = False
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Membangun Matriks Kookurensi X(i, j).
    
    Argumen:
    - tokenized_corpus: daftar kalimat yang telah ditokenisasi, misal [['barang', 'bagus', 'sekali'], ...]
    - vocab: daftar kata kosakata. Jika None, diambil dari kata unik di korpus sesuai urutan kemunculan.
    - window_size: lebar jendela konteks ke kiri dan kanan (default 1 untuk contoh proposal, 5 untuk nyata).
    - distance_weighting: jika True, bobot = 1 / jarak. Jika False, bobot = 1.0 (sesuai simulasi manual 4.1).
    
    Return:
    - vocab: daftar kosakata kata unik
    - matrix: numpy array 2D berukuran (V, V)
    - df_matrix: pandas DataFrame berlabel baris & kolom kata
    """
    if vocab is None:
        seen = set()
        vocab_list = []
        for doc in tokenized_corpus:
            for w in doc:
                if w not in seen:
                    seen.add(w)
                    vocab_list.append(w)
        vocab = vocab_list

    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    X = np.zeros((V, V), dtype=np.float64)

    for doc in tokenized_corpus:
        n = len(doc)
        for i, center_word in enumerate(doc):
            if center_word not in word2idx:
                continue
            center_idx = word2idx[center_word]

            # Jendela konteks
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue
                context_word = doc[j]
                if context_word not in word2idx:
                    continue
                context_idx = word2idx[context_word]

                dist = abs(i - j)
                weight = (1.0 / dist) if distance_weighting else 1.0

                X[center_idx, context_idx] += weight

    df_matrix = pd.DataFrame(X, index=vocab, columns=vocab)
    return vocab, X, df_matrix

