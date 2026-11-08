"""
Unit Testing untuk Engine SentiGlove
Memvalidasi:
1. Preprocessor (Case folding, slang, stopword negasi)
2. GloVe Loader (Vector Averaging 50d, OOV zero vector)
3. Matriks Kookurensi (Replikasi Tabel 3 Proposal Bab 4)
4. Simulasi GloVe SGD (Replikasi Tabel 4 Proposal Bab 4)
5. Klasifikasi Sentimen Centroid (Replikasi Tabel 5 Proposal Bab 4)
6. Model SentiGlove Classifier
"""

import unittest
import numpy as np
from engine.preprocessor import clean_text, tokenize, normalize_slang, remove_stopwords, preprocess_review
from engine.glove_loader import GloVeEmbedding
from engine.cooccurrence import build_cooccurrence_matrix
from engine.simulation import glove_step_calculation, cosine_similarity, get_proposal_simulation_data
from engine.classifier import SentiGloveClassifier


class TestSentiGloveEngine(unittest.TestCase):

    def test_preprocessor(self):
        # Uji normalisasi slang dan negasi
        raw = "Barang bgs bgt!! Pengiriman cepet, tdk kecewa :)"
        tokens = preprocess_review(raw)
        self.assertIn("bagus", tokens)
        self.assertIn("banget", tokens)
        self.assertIn("cepat", tokens)
        self.assertIn("tidak", tokens)  # Negasi tidak boleh dihapus
        self.assertIn("kecewa", tokens)

    def test_proposal_table_3_cooccurrence(self):
        # Replikasi Bab 4.1 Tabel 3 Proposal
        corpus = [
            ["barang", "bagus", "sekali"],
            ["barang", "jelek", "sekali"],
            ["pengiriman", "cepat", "barang", "bagus"]
        ]
        vocab_order = ["barang", "bagus", "jelek", "sekali", "cepat", "pengiriman"]
        vocab, X, df_X = build_cooccurrence_matrix(
            corpus, vocab=vocab_order, window_size=1, distance_weighting=False
        )

        word2idx = {w: i for i, w in enumerate(vocab)}
        # X(barang, bagus) = 2
        self.assertEqual(X[word2idx["barang"], word2idx["bagus"]], 2.0)
        # X(bagus, barang) = 2
        self.assertEqual(X[word2idx["bagus"], word2idx["barang"]], 2.0)
        # X(barang, jelek) = 1
        self.assertEqual(X[word2idx["barang"], word2idx["jelek"]], 1.0)
        # X(barang, sekali) = 0
        self.assertEqual(X[word2idx["barang"], word2idx["sekali"]], 0.0)
        # X(cepat, pengiriman) = 1
        self.assertEqual(X[word2idx["cepat"], word2idx["pengiriman"]], 1.0)

    def test_proposal_table_4_sgd_step(self):
        # Replikasi Bab 4.2 Tabel 4 Proposal
        res = glove_step_calculation(
            x_ij=2.0,
            w_i=np.array([0.30, 0.10]),
            w_tilde_j=np.array([0.20, 0.40]),
            b_i=0.10,
            b_tilde_j=0.05,
            x_max=10.0,
            alpha=0.75,
            eta=0.1
        )

        # Cek nilai-nilai kunci sesuai tabel proposal
        self.assertAlmostEqual(res["log_x"], 0.6931, places=3)
        self.assertAlmostEqual(res["f_val"], 0.2991, places=3)
        self.assertAlmostEqual(res["pred"], 0.2500, places=3)
        self.assertAlmostEqual(res["d"], -0.4431, places=3)
        self.assertAlmostEqual(res["J"], 0.0587, places=3)
        self.assertAlmostEqual(res["g"], -0.2651, places=3)
        self.assertAlmostEqual(res["w_i_new"][0], 0.3053, places=3)
        self.assertAlmostEqual(res["w_i_new"][1], 0.1106, places=3)
        self.assertAlmostEqual(res["w_tilde_j_new"][0], 0.2080, places=3)
        self.assertAlmostEqual(res["w_tilde_j_new"][1], 0.4027, places=3)
        self.assertAlmostEqual(res["b_i_new"], 0.1265, places=3)
        self.assertAlmostEqual(res["b_tilde_j_new"], 0.0765, places=3)
        self.assertAlmostEqual(res["pred_new"], 0.3110, places=3)
        self.assertAlmostEqual(res["d_new"], -0.3821, places=3)
        self.assertAlmostEqual(res["J_new"], 0.0437, places=3)
        # Loss harus menurun
        self.assertGreater(res["loss_reduction"], 0.0)

    def test_proposal_table_5_classification(self):
        # Replikasi Bab 4.3 Tabel 5 Proposal
        sim_data = get_proposal_simulation_data()
        r_list = sim_data["review_results"]

        r1 = r_list[0]
        self.assertEqual(r1["text"], "barang cepat bagus")
        self.assertEqual(r1["r_vec"], [0.450, 0.300])
        self.assertAlmostEqual(r1["cos_pos"], 0.998, places=3)
        self.assertAlmostEqual(r1["cos_neg"], -0.995, places=3)
        self.assertEqual(r1["pred"], "Positif")

        r2 = r_list[1]
        self.assertEqual(r2["text"], "barang jelek sekali")
        self.assertEqual(r2["r_vec"], [-0.050, -0.100])
        self.assertAlmostEqual(r2["cos_pos"], -0.898, places=3)
        self.assertAlmostEqual(r2["cos_neg"], 0.913, places=3)
        self.assertEqual(r2["pred"], "Negatif")

    def test_glove_embedding_vector_averaging(self):
        glove = GloVeEmbedding.get_instance()
        self.assertEqual(glove.dimension, 50)

        vec, info = glove.sentence_vector(["bagus", "cepat", "asli"])
        self.assertEqual(vec.shape, (50,))
        self.assertGreater(np.linalg.norm(vec), 0.0)

        # Kata OOV
        vec_oov, info_oov = glove.sentence_vector(["kataanehyangtidakmungkinada12345"])
        self.assertEqual(info_oov["oov_count"], 1)
        self.assertAlmostEqual(float(np.linalg.norm(vec_oov)), 0.0)

    def test_classifier_predictions(self):
        classifier = SentiGloveClassifier.get_instance()
        self.assertTrue(classifier.is_trained)

        # Ulasan positif
        res_pos = classifier.predict_text("Barang sangat bagus, kualitas original, packing aman dan cepat sekali sampai!")
        self.assertEqual(res_pos["label"], "Positif")
        self.assertGreater(res_pos["skor_pos"], res_pos["skor_neg"])

        # Ulasan negatif
        res_neg = classifier.predict_text("Barang jelek, rusak, pecah, seller lambat tidak ada respon dan mengecewakan")
        self.assertEqual(res_neg["label"], "Negatif")
        self.assertGreater(res_neg["skor_neg"], res_pos["skor_neg"])


if __name__ == "__main__":
    unittest.main()

