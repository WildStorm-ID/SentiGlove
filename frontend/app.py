"""
Aplikasi Web SentiGlove Dashboard (Streamlit)
Analisis Sentimen Ulasan Produk E-Commerce Berbasis Word Embedding GloVe
"""

import os
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.db import SessionLocal, engine, Base
from database.models import Penjual, Produk, Ulasan, ModelGlove, VektorKata, Kookurensi, HasilSentimen, get_utc_now
from engine.classifier import SentiGloveClassifier, cosine_sim
from engine.preprocessor import clean_text, tokenize, normalize_slang, remove_stopwords, preprocess_review
from engine.cooccurrence import build_cooccurrence_matrix
from engine.simulation import glove_step_calculation, get_proposal_simulation_data, cosine_similarity
from frontend.components import (
    apply_custom_styles,
    render_metric_card,
    sentiment_donut_chart,
    category_sentiment_bar_chart,
    vector_dimension_bar_chart
)

# Inisialisasi konfigurasi halaman Streamlit
st.set_page_config(
    page_title="SentiGlove - Analisis Sentimen Ulasan E-Commerce",
    page_icon="🧤",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles()


# Dependency helper untuk sesi basis data
def get_db_session():
    return SessionLocal()


# Cache resource untuk classifier agar cepat
@st.cache_resource
def load_classifier():
    return SentiGloveClassifier.get_instance()


classifier = load_classifier()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 🧤 **SentiGlove**")
    st.caption("Analisis Sentimen Ulasan E-Commerce Berbasis Word Embedding GloVe")
    st.divider()

    menu = st.radio(
        "Navigasi Modul:",
        [
            "📊 Dashboard Penjual",
            "⚡ Uji Sentimen Real-Time",
            "🧮 Simulasi Matematis GloVe",
            "📁 Import CSV & Basis Data",
            "🏛️ Perancangan Sistem (DFD & ERD)"
        ],
        index=0
    )

    st.divider()
    st.markdown("### ⚙️ Status Sistem")
    st.success("✅ Model GloVe 50D Aktif")
    st.info("📦 Korpus: Wiki ID + Tokopedia")
    st.caption("UTS Natural Language Processing\nStudi Kasus Word Embedding GloVe")


# ==========================================
# 1. DASHBOARD PENJUAL
# ==========================================
if menu == "📊 Dashboard Penjual":
    st.title("📊 Dashboard Analisis Sentimen Penjual")
    st.write(
        "Ringkasan analitik ulasan produk marketplace berdasarkan representasi vektor kata GloVe dan klasifikasi sentimen biner (Positif vs Negatif)."
    )

    db = get_db_session()
    try:
        # Metrik Utama
        total_penjual = db.query(Penjual).count()
        total_produk = db.query(Produk).count()
        total_ulasan = db.query(Ulasan).count()
        total_pos = db.query(HasilSentimen).filter_by(label_prediksi="Positif").count()
        total_neg = db.query(HasilSentimen).filter_by(label_prediksi="Negatif").count()

        avg_rating = db.query(func.avg(Ulasan.rating_asli)).scalar() or 0.0
        pct_pos = round((total_pos / total_ulasan * 100), 1) if total_ulasan > 0 else 0.0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_metric_card("Total Toko", f"{total_penjual}", "Toko terdaftar")
        with col2:
            render_metric_card("Total Produk", f"{total_produk}", "Katalog produk")
        with col3:
            render_metric_card("Total Ulasan", f"{total_ulasan:,}", "Ulasan diproses")
        with col4:
            render_metric_card("Sentimen Positif", f"{pct_pos}%", f"{total_pos} Positif / {total_neg} Negatif")
        with col5:
            render_metric_card("Rating Rata-rata", f"{avg_rating:.2f} ⭐", "Skala 1 - 5")

        st.markdown("---")

        # Filter Interaktif
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            kategori_list = ["Semua Kategori"] + [
                k[0] for k in db.query(Produk.kategori).distinct().all() if k[0]
            ]
            sel_kategori = st.selectbox("Pilih Kategori:", kategori_list)

        with col_f2:
            penjual_list = ["Semua Toko"] + [
                f"{p.id_penjual} - {p.nama_toko}" for p in db.query(Penjual).all()
            ]
            sel_penjual = st.selectbox("Pilih Toko / Penjual:", penjual_list)

        with col_f3:
            prod_query = db.query(Produk)
            if sel_kategori != "Semua Kategori":
                prod_query = prod_query.filter(Produk.kategori == sel_kategori)
            if sel_penjual != "Semua Toko":
                p_id = sel_penjual.split(" - ")[0]
                prod_query = prod_query.filter(Produk.id_penjual == p_id)
            produk_list = ["Semua Produk"] + [
                f"{p.id_produk} - {p.nama_produk[:35]}..." for p in prod_query.limit(100).all()
            ]
            sel_produk = st.selectbox("Pilih Produk Spesifik:", produk_list)

        # Filter Query Ulasan
        ulasan_query = db.query(Ulasan).join(HasilSentimen, Ulasan.id_ulasan == HasilSentimen.id_ulasan)
        if sel_kategori != "Semua Kategori":
            ulasan_query = ulasan_query.join(Produk, Ulasan.id_produk == Produk.id_produk).filter(
                Produk.kategori == sel_kategori
            )
        if sel_penjual != "Semua Toko":
            p_id = sel_penjual.split(" - ")[0]
            ulasan_query = ulasan_query.join(Produk, Ulasan.id_produk == Produk.id_produk).filter(
                Produk.id_penjual == p_id
            )
        if sel_produk != "Semua Produk":
            prod_id_target = sel_produk.split(" - ")[0]
            ulasan_query = ulasan_query.filter(Ulasan.id_produk == prod_id_target)

        filtered_reviews = ulasan_query.all()
        f_pos = sum(1 for u in filtered_reviews if u.hasil_sentimen.label_prediksi == "Positif")
        f_neg = sum(1 for u in filtered_reviews if u.hasil_sentimen.label_prediksi == "Negatif")

        st.markdown("### 📈 Visualisasi Sentimen & Rating")
        c_chart1, c_chart2 = st.columns([1, 2])

        with c_chart1:
            st.markdown("#### Proporsi Sentimen")
            st.altair_chart(sentiment_donut_chart(f_pos, f_neg), use_container_width=True)

        with c_chart2:
            st.markdown("#### Distribusi Rating Asli Ulasan")
            ratings = [u.rating_asli for u in filtered_reviews if u.rating_asli is not None]
            if ratings:
                df_ratings = pd.DataFrame({"Rating": [f"Bintang {r}" for r in ratings]}).value_counts().reset_index()
                df_ratings.columns = ["Rating", "Jumlah"]
                chart_rat = alt.Chart(df_ratings).mark_bar(color="#38bdf8", cornerRadius=4).encode(
                    x=alt.X("Rating:N", sort=["Bintang 1", "Bintang 2", "Bintang 3", "Bintang 4", "Bintang 5"]),
                    y="Jumlah:Q",
                    tooltip=["Rating", "Jumlah"]
                ).properties(height=280)
                st.altair_chart(chart_rat, use_container_width=True)
            else:
                st.info("Tidak ada data rating pada filter ini.")

        # Perbandingan Kategori
        st.markdown("### 📦 Distribusi Sentimen per Kategori")
        cat_stats = []
        for cat_name in [k[0] for k in db.query(Produk.kategori).distinct().all() if k[0]]:
            q_cat = db.query(HasilSentimen).join(Ulasan, HasilSentimen.id_ulasan == Ulasan.id_ulasan).join(
                Produk, Ulasan.id_produk == Produk.id_produk
            ).filter(Produk.kategori == cat_name)
            p_cnt = q_cat.filter(HasilSentimen.label_prediksi == "Positif").count()
            n_cnt = q_cat.filter(HasilSentimen.label_prediksi == "Negatif").count()
            cat_stats.append({"kategori": cat_name, "sentimen": "Positif", "jumlah": p_cnt})
            cat_stats.append({"kategori": cat_name, "sentimen": "Negatif", "jumlah": n_cnt})

        if cat_stats:
            df_cat = pd.DataFrame(cat_stats)
            st.altair_chart(category_sentiment_bar_chart(df_cat), use_container_width=True)

        # Tabel Ulasan Terkini
        st.markdown("### 💬 Daftar Ulasan Produk")
        table_rows = []
        for u in filtered_reviews[:50]:
            hs = u.hasil_sentimen
            prod_name = u.produk.nama_produk if u.produk else u.id_produk
            table_rows.append({
                "ID": u.id_ulasan,
                "Produk": prod_name[:30] + "...",
                "Teks Ulasan": u.teks,
                "Rating": f"{u.rating_asli} ⭐" if u.rating_asli else "-",
                "Sentimen": hs.label_prediksi if hs else "-",
                "Skor Positif (cos c+)": round(hs.skor_pos, 4) if hs else 0.0,
                "Skor Negatif (cos c-)": round(hs.skor_neg, 4) if hs else 0.0,
            })

        if table_rows:
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, height=350)
        else:
            st.warning("Tidak ada ulasan ditemukan untuk filter terpilih.")

    finally:
        db.close()


# ==========================================
# 2. UJI SENTIMEN REAL-TIME
# ==========================================
elif menu == "⚡ Uji Sentimen Real-Time":
    st.title("⚡ Pengujian Sentimen Ulasan Real-Time")
    st.write(
        "Ketikkan teks ulasan baru untuk melihat langsung klasifikasi sentimen menggunakan pipeline GloVe 50D, "
        "Vector Averaging, dan Cosine Similarity ke pusat kelas (c+ vs c-)."
    )

    # Preset contoh kalimat
    col_preset, _ = st.columns([4, 1])
    with col_preset:
        st.markdown("**Gunakan Contoh Ulasan Cepat:**")
        p1, p2, p3, p4 = st.columns(4)
        sample_text = ""
        if p1.button("✨ Ulasan Sangat Puas"):
            st.session_state["review_input"] = "Barang original, sangat bagus dan berkualitas! Respon seller kilat, packing rapih aman."
        if p2.button("⚠️ Ulasan Sangat Kecewa"):
            st.session_state["review_input"] = "Barang jelek, rusak pecah parah, seller lambat tidak ada respon dan sangat mengecewakan!"
        if p3.button("📦 Ulasan Singkat Positif"):
            st.session_state["review_input"] = "Mantap bos, barang sampai cepat dan sesuai pesanan. Recommended seller."
        if p4.button("❌ Ulasan Singkat Negatif"):
            st.session_state["review_input"] = "Barang yang dikirim tidak sesuai gambar dan cacat, kapok belanja di sini."

    review_text = st.text_area(
        "Teks Ulasan Pembeli:",
        value=st.session_state.get("review_input", "Barang bagus harga murah, packing aman mantap cepat sampai!"),
        height=100
    )

    col_btn, col_opt = st.columns([1, 2])
    with col_btn:
        btn_predict = st.button("🚀 Analisis Sentimen Ulasan", type="primary", use_container_width=True)
    with col_opt:
        save_to_db = st.checkbox("Simpan ulasan dan hasil prediksi ke Database SQLite", value=False)

    if btn_predict and review_text.strip():
        with st.spinner("Memproses praproses teks, ekstraksi vektor GloVe 50D, dan menghitung Cosine Similarity..."):
            res = classifier.predict_text(review_text)

        st.markdown("---")
        st.subheader("🎯 Hasil Klasifikasi Sentimen")

        # Kartu Hasil Utama
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            if res["label"] == "Positif":
                st.markdown("<div class='badge-pos' style='font-size:1.4rem; padding: 10px 20px;'>POSITIF 👍</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge-neg' style='font-size:1.4rem; padding: 10px 20px;'>NEGATIF 👎</div>", unsafe_allow_html=True)
            st.caption(f"Tingkat Keyakinan: {res['confidence']}%")

        with col_res2:
            render_metric_card("Cosine Sim c+ (Positif)", f"{res['skor_pos']:.4f}", "Pusat Kelas Positif")

        with col_res3:
            render_metric_card("Cosine Sim c- (Negatif)", f"{res['skor_neg']:.4f}", "Pusat Kelas Negatif")

        with col_res4:
            render_metric_card("Fallback Logistik", res["logistic_label"], f"P(Pos): {res['logistic_prob_pos']:.2f} | P(Neg): {res['logistic_prob_neg']:.2f}")

        # Simpan ke DB jika dicentang
        if save_to_db:
            db = get_db_session()
            try:
                first_prod = db.query(Produk).first()
                p_id = first_prod.id_produk if first_prod else "prod_live_test"
                u_obj = Ulasan(
                    id_produk=p_id,
                    teks=review_text,
                    rating_asli=5 if res["label"] == "Positif" else 1,
                    label_asli=res["label"],
                    created_at=get_utc_now()
                )
                db.add(u_obj)
                db.flush()
                hs_obj = HasilSentimen(
                    id_ulasan=u_obj.id_ulasan,
                    id_model="glove_wiki_id_50d",
                    skor_pos=res["skor_pos"],
                    skor_neg=res["skor_neg"],
                    label_prediksi=res["label"],
                    created_at=get_utc_now()
                )
                db.add(hs_obj)
                db.commit()
                st.success(f"✅ Ulasan berhasil disimpan ke database dengan ID Ulasan #{u_obj.id_ulasan}!")
            except Exception as e:
                db.rollback()
                st.error(f"Gagal menyimpan ke database: {e}")
            finally:
                db.close()

        # Rincian Teknis Pipeline
        st.markdown("### 🔍 Rincian Tahapan NLP (Proses 1.0 - 4.0 DFD)")

        with st.expander("1. Tahap Praproses Teks (Case folding, Slang Normalization, Stopwords)", expanded=True):
            clean_str = clean_text(review_text)
            st.markdown(f"**Teks Asli:** `{review_text}`")
            st.markdown(f"**Pembersihan Non-Alfabet:** `{clean_str}`")
            st.markdown(f"**Token Akhir Bersih:** `{res['tokens']}`")
            st.info(
                f"📊 Total Kata: {res['token_info']['total_tokens']} | "
                f"Kata Ada di Kamus GloVe: {res['token_info']['matched_count']} | "
                f"Kata OOV (Vektor Nol): {res['token_info']['oov_count']} | "
                f"Cakupan Kosakata: {res['token_info']['coverage_pct']}%"
            )
            if res["token_info"]["oov_words"]:
                st.caption(f"Kata OOV: {', '.join(res['token_info']['oov_words'])}")

        with st.expander("2. Representasi Vector Average 50-Dimensi (GloVe Embedding)", expanded=True):
            st.write("Vektor ulasan dihitung sebagai rata-rata koordinat tiap kata: $r = \\frac{1}{|V|} \\sum_{w \\in V} v(w)$")
            st.altair_chart(vector_dimension_bar_chart(res["vector"]), use_container_width=True)
            st.caption("Nilai Vektor (Array 50 Elemen Float):")
            st.code(str(res["vector"]), language="json")


# ==========================================
# 3. SIMULASI MATEMATIS GLOVE (BAB 4 PROPOSAL)
# ==========================================
elif menu == "🧮 Simulasi Matematis GloVe":
    st.title("🧮 Simulasi Perhitungan Matematis GloVe")
    st.write(
        "Modul interaktif ini memvalidasi dan mereplikasi seluruh simulasi perhitungan manual."
    )

    tab1, tab2, tab3 = st.tabs([
        "Matriks Kookurensi X",
        "Fungsi Bobot, Loss, & Pembaruan SGD",
        "Klasifikasi Sentimen Centroid (c+ vs c-)"
    ])

    # TAB 1: Matriks Kookurensi
    with tab1:
        st.subheader("Matriks Kookurensi X(i, j)")
        st.write(
            "Mencatat kemunculan bersama kata $i$ dan kata konteks $j$ dalam jendela simetris korpus."
        )

        st.markdown("""
        **Korpus Tiga Ulasan:**
        - $S_1$: *“barang bagus sekali”*
        - $S_2$: *“barang jelek sekali”*
        - $S_3$: *“pengiriman cepat barang bagus”*
        """)

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            win_size = st.slider("Ukuran Jendela (Window Size):", min_value=1, max_value=5, value=1)
        with c_opt2:
            use_dist = st.checkbox("Gunakan Pembobotan Jarak (1 / Jarak)", value=False)

        custom_corpus = st.text_area(
            "Ubah Korpus Kalimat (1 kalimat per baris):",
            value="barang bagus sekali\nbarang jelek sekali\npengiriman cepat barang bagus",
            height=150
        )

        tokenized_lines = [line.strip().split() for line in custom_corpus.strip().split("\n") if line.strip()]
        vocab, X, df_X = build_cooccurrence_matrix(
            tokenized_lines, window_size=win_size, distance_weighting=use_dist
        )

        st.markdown("#### Tabel Matriks Kookurensi $X(i, j)$:")
        st.dataframe(df_X.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

        # if win_size == 1 and not use_dist:
        #     st.success("✅ Nilai matriks di atas persis sesuai dengan **Tabel 3 Proposal UTS NLP**!")

    # TAB 2: Bobot, Loss, SGD
    with tab2:
        st.subheader("Fungsi Bobot, Loss GloVe, dan Pembaruan SGD (1 Iterasi)")
        st.write(
            "Minimasi fungsi loss kuadrat berbobot: "
            "$$J = \\sum_{i,j} f(X_{ij}) \\cdot (w_i \\cdot \\tilde{w}_j + b_i + \\tilde{b}_j - \\log X_{ij})^2$$"
        )

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            x_val = st.number_input("Frekuensi Kookurensi $X_{ij}$:", value=2.0, min_value=0.1, step=0.5)
            xmax_val = st.number_input("Parameter $x_{max}$:", value=10.0, min_value=1.0, step=1.0)
        with col_p2:
            alpha_val = st.number_input("Eksponen $\\alpha$:", value=0.75, min_value=0.1, max_value=1.0, step=0.05)
            eta_val = st.number_input("Learning Rate $\\eta$:", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
        with col_p3:
            st.markdown("**Nilai Awal Parameter (2D):**")
            st.caption("w_barang = [0.30, 0.10], b = 0.10")
            st.caption("w~_bagus = [0.20, 0.40], b~ = 0.05")

        # Hitung satu langkah SGD
        step_calc = glove_step_calculation(
            x_ij=x_val,
            w_i=np.array([0.30, 0.10]),
            w_tilde_j=np.array([0.20, 0.40]),
            b_i=0.10,
            b_tilde_j=0.05,
            x_max=xmax_val,
            alpha=alpha_val,
            eta=eta_val
        )

        st.markdown("#### Tabel Perhitungan Manual Langkah Demi Langkah:")

        step_rows = [
            {"Langkah": "1. Data", "Rumus Matematis": "Xij dari matriks; log Xij", "Hasil Perhitungan": f"X = {step_calc['x_ij']}; log({step_calc['x_ij']}) = {step_calc['log_x']}"},
            {"Langkah": "2. Bobot", "Rumus Matematis": "f(x) = (x / xmax)^alpha", "Hasil Perhitungan": f"f({step_calc['x_ij']}) = ({step_calc['x_ij']}/{xmax_val})^{alpha_val} = {step_calc['f_val']}"},
            {"Langkah": "3. Prediksi", "Rumus Matematis": "w·w~ + b + b~", "Hasil Perhitungan": f"dot_prod ({step_calc['dot_product']}) + {step_calc['b_i_old']} + {step_calc['b_tilde_j_old']} = {step_calc['pred']}"},
            {"Langkah": "4. Selisih", "Rumus Matematis": "d = prediksi - log X", "Hasil Perhitungan": f"d = {step_calc['pred']} - {step_calc['log_x']} = {step_calc['d']}"},
            {"Langkah": "5. Loss Awal", "Rumus Matematis": "J = f · d^2", "Hasil Perhitungan": f"J = {step_calc['f_val']} × ({step_calc['d']})^2 = {step_calc['J']}"},
            {"Langkah": "6. Gradien", "Rumus Matematis": "g = 2·f·d; ∂J/∂w = g·w~; ∂J/∂w~ = g·w; ∂J/∂b = g", "Hasil Perhitungan": f"g = {step_calc['g']}; ∂J/∂w = {step_calc['grad_w_i']}; ∂J/∂w~ = {step_calc['grad_w_tilde_j']}"},
            {"Langkah": "7. Pembaruan", "Rumus Matematis": "theta <- theta - eta · ∂J/∂theta", "Hasil Perhitungan": f"w = {step_calc['w_i_new']}; w~ = {step_calc['w_tilde_j_new']}; b = {step_calc['b_i_new']}; b~ = {step_calc['b_tilde_j_new']}"},
            {"Langkah": "8. Cek Ulang", "Rumus Matematis": "d_baru; J_baru = f · (d_baru)^2", "Hasil Perhitungan": f"d_baru = {step_calc['d_new']}; J_baru = {step_calc['J_new']} (Penurunan loss: {step_calc['loss_reduction']})"}
        ]
        st.table(pd.DataFrame(step_rows))

        # if x_val == 2.0 and xmax_val == 10.0 and alpha_val == 0.75 and eta_val == 0.1:
        #     st.success("✅ Seluruh angka perhitungan langkah 1 s/d 8 di atas persis sesuai dengan **Tabel 4 Proposal UTS NLP**!")

    # TAB 3: Klasifikasi Sentimen Proposal
    with tab3:
        st.subheader("Klasifikasi Sentimen Ulasan Baru")
        st.write(
            "Menggunakan vektor kata hasil pelatihan 2D untuk menghitung rata-rata vektor ulasan $r$, "
            "lalu mengukur Cosine Similarity terhadap pusat kelas positif ($c^+$) dan negatif ($c^-$)."
        )

        sim_prop = get_proposal_simulation_data()
        st.markdown(f"""
        - **Pusat Kelas Positif ($c^+$):** `rata-rata(bagus, cepat)` = `{sim_prop['c_pos']}`
        - **Pusat Kelas Negatif ($c^-$):** `vektor jelek` = `{sim_prop['c_neg']}`
        """)

        # Tampilkan Tabel 5
        st.markdown("#### Tabel Hasil Klasifikasi Ulasan Baru:")
        df_t5 = pd.DataFrame(sim_prop["review_results"])
        df_t5.columns = ["Ulasan Baru", "Vektor Rata-rata r", "cos(r, c+)", "cos(r, c-)", "Prediksi"]
        st.table(df_t5)

        # Visualisasi 2D Vektor Kata & Centroid
        st.markdown("#### 🗺️ Peta Visual Ruang Vektor 2D:")
        plot_data = []
        for word, coord in sim_prop["word_vectors_2d"].items():
            plot_data.append({"Entity": f"Kata: {word}", "X": coord[0], "Y": coord[1], "Tipe": "Kata"})
        plot_data.append({"Entity": "Centroid c+ (Positif)", "X": sim_prop["c_pos"][0], "Y": sim_prop["c_pos"][1], "Tipe": "Centroid Positif"})
        plot_data.append({"Entity": "Centroid c- (Negatif)", "X": sim_prop["c_neg"][0], "Y": sim_prop["c_neg"][1], "Tipe": "Centroid Negatif"})
        for r in sim_prop["review_results"]:
            plot_data.append({"Entity": f"Ulasan: '{r['text']}'", "X": r["r_vec"][0], "Y": r["r_vec"][1], "Tipe": f"Ulasan {r['pred']}"})

        df_plot = pd.DataFrame(plot_data)

        scatter_chart = alt.Chart(df_plot).mark_circle(size=140).encode(
            x=alt.X("X:Q", title="Dimensi 1"),
            y=alt.Y("Y:Q", title="Dimensi 2"),
            color=alt.Color("Tipe:N", scale=alt.Scale(
                domain=["Kata", "Centroid Positif", "Centroid Negatif", "Ulasan Positif", "Ulasan Negatif"],
                range=["#94a3b8", "#10b981", "#ef4444", "#38bdf8", "#f43f5e"]
            )),
            tooltip=["Entity", "X", "Y", "Tipe"]
        ).properties(
            width=700,
            height=380
        )

        text_labels = alt.Chart(df_plot).mark_text(dy=-12, fontSize=11).encode(
            x="X:Q",
            y="Y:Q",
            text="Entity:N"
        )

        st.altair_chart(scatter_chart + text_labels, use_container_width=True)


# ==========================================
# 4. IMPORT CSV & BASIS DATA
# ==========================================
elif menu == "📁 Import CSV & Basis Data":
    st.title("📁 Import CSV & Penjelajah Basis Data Relasional")
    st.write(
        "Kelola basis data relasional SQLite SentiGlove dan impor berkas CSV ulasan produk baru."
    )

    tab_import, tab_db = st.tabs(["📤 Form Import File CSV", "🗄️ Penjelajah Basis Data (ERD 6 Entitas)"])

    with tab_import:
        st.subheader("Unggah File CSV Ulasan Produk")
        st.write(
            "Unggah file CSV ulasan produk (misal ulasan Tokopedia/Shopee). "
            "Sistem akan memvalidasi kolom, melakukan praproses, dan mengklasifikasikan sentimen secara otomatis."
        )

        uploaded_file = st.file_uploader("Pilih Berkas CSV:", type=["csv"])
        max_import_rows = st.slider("Batas Jumlah Baris yang Diimpor:", min_value=10, max_value=500, value=50, step=10)

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Pratinjau Data Unggahan ({len(df_upload)} baris ditemukan):")
                st.dataframe(df_upload.head(5), use_container_width=True)

                if st.button("📥 Proses & Simpan ke Basis Data", type="primary"):
                    db = get_db_session()
                    try:
                        # Identifikasi kolom
                        text_col = next((c for c in ["text", "ulasan", "review"] if c in df_upload.columns), None)
                        rating_col = next((c for c in ["rating", "bintang", "score"] if c in df_upload.columns), None)

                        if not text_col:
                            st.error("Kolom ulasan tidak ditemukan. Pastikan ada kolom bernama 'text' atau 'ulasan'.")
                        else:
                            # Filter rating 3
                            df_proc = df_upload.copy()
                            if rating_col:
                                df_proc = df_proc[df_proc[rating_col] != 3]

                            df_proc = df_proc.head(max_import_rows)

                            penjual_cache = {p[0] for p in db.query(Penjual.id_penjual).all()}
                            produk_cache = {p[0] for p in db.query(Produk.id_produk).all()}

                            count_pos = 0
                            count_neg = 0
                            imported = 0

                            progress_bar = st.progress(0)
                            total_rows = len(df_proc)

                            for i, (_, row) in enumerate(df_proc.iterrows()):
                                t = str(row[text_col]).strip()
                                if not t or t.lower() == "nan":
                                    continue

                                rat = int(row[rating_col]) if rating_col and pd.notnull(row[rating_col]) else None
                                lbl_asli = ("Positif" if rat >= 4 else "Negatif") if rat else None

                                s_id = str(row.get("shop_id", "toko_csv_import"))
                                p_id = str(row.get("product_id", f"prod_csv_{i}"))
                                p_name = str(row.get("product_name", f"Produk CSV {i}"))
                                cat = str(row.get("category", "Umum"))

                                if s_id not in penjual_cache:
                                    db.add(Penjual(id_penjual=s_id, nama_toko=f"Toko #{s_id}"))
                                    db.flush()
                                    penjual_cache.add(s_id)

                                if p_id not in produk_cache:
                                    db.add(Produk(id_produk=p_id, id_penjual=s_id, nama_produk=p_name, kategori=cat))
                                    db.flush()
                                    produk_cache.add(p_id)

                                u_obj = Ulasan(
                                    id_produk=p_id,
                                    teks=t,
                                    rating_asli=rat,
                                    label_asli=lbl_asli,
                                    created_at=get_utc_now()
                                )
                                db.add(u_obj)
                                db.flush()

                                # Klasifikasi SentiGlove
                                pred = classifier.predict_text(t)
                                if pred["label"] == "Positif":
                                    count_pos += 1
                                else:
                                    count_neg += 1

                                db.add(HasilSentimen(
                                    id_ulasan=u_obj.id_ulasan,
                                    id_model="glove_wiki_id_50d",
                                    skor_pos=pred["skor_pos"],
                                    skor_neg=pred["skor_neg"],
                                    label_prediksi=pred["label"],
                                    created_at=get_utc_now()
                                ))
                                imported += 1
                                progress_bar.progress((i + 1) / total_rows)

                            db.commit()
                            st.success(
                                f"🎉 Berhasil mengimpor {imported} ulasan! "
                                f"Hasil Sentimen: {count_pos} Positif, {count_neg} Negatif."
                            )
                    except Exception as ex:
                        db.rollback()
                        st.error(f"Gagal saat proses impor: {ex}")
                    finally:
                        db.close()
            except Exception as e:
                st.error(f"Gagal membaca berkas CSV: {e}")

    with tab_db:
        st.subheader("Penjelajah Tabel Basis Data Relasional")
        selected_table = st.selectbox(
            "Pilih Tabel untuk Dilihat:",
            ["PENJUAL", "PRODUK", "ULASAN", "HASIL_SENTIMEN", "MODEL_GLOVE", "VEKTOR_KATA", "KOOKURENSI"]
        )

        db = get_db_session()
        try:
            if selected_table == "PENJUAL":
                data = [{"ID Penjual": p.id_penjual, "Nama Toko": p.nama_toko, "Jumlah Produk": len(p.produks)} for p in db.query(Penjual).all()]
            elif selected_table == "PRODUK":
                data = [{"ID Produk": p.id_produk, "ID Penjual": p.id_penjual, "Nama Produk": p.nama_produk, "Kategori": p.kategori, "Jumlah Ulasan": len(p.ulasans)} for p in db.query(Produk).limit(200).all()]
            elif selected_table == "ULASAN":
                data = [{"ID Ulasan": u.id_ulasan, "ID Produk": u.id_produk, "Teks": u.teks[:60] + "...", "Rating": u.rating_asli, "Label Asli": u.label_asli} for u in db.query(Ulasan).limit(200).all()]
            elif selected_table == "HASIL_SENTIMEN":
                data = [{"ID Hasil": h.id_hasil, "ID Ulasan": h.id_ulasan, "ID Model": h.id_model, "Skor Pos": h.skor_pos, "Skor Neg": h.skor_neg, "Label Prediksi": h.label_prediksi} for h in db.query(HasilSentimen).limit(200).all()]
            elif selected_table == "MODEL_GLOVE":
                data = [{"ID Model": m.id_model, "Dimensi": m.dimensi, "x_max": m.x_max, "alpha": m.alpha, "Loss Akhir": m.loss_akhir, "Keterangan": m.keterangan} for m in db.query(ModelGlove).all()]
            elif selected_table == "VEKTOR_KATA":
                data = [{"ID Model": v.id_model, "Kata": v.kata, "Vektor": v.vektor_w[:30] + "...", "Bias b": v.bias_b} for v in db.query(VektorKata).all()]
            elif selected_table == "KOOKURENSI":
                data = [{"ID Model": k.id_model, "Kata i": k.kata_i, "Kata j": k.kata_j, "Jumlah X(i,j)": k.jumlah} for k in db.query(Kookurensi).all()]

            if data:
                df_out = pd.DataFrame(data)
                st.dataframe(df_out, use_container_width=True)
                st.caption(f"Menampilkan {len(df_out)} baris data.")
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"💾 Unduh Tabel {selected_table} (.csv)",
                    data=csv_bytes,
                    file_name=f"{selected_table.lower()}_export.csv",
                    mime="text/csv"
                )
            else:
                st.info("Tabel ini masih kosong.")
        finally:
            db.close()


# ==========================================
# 5. PERANCANGAN SISTEM (DFD & ERD)
# ==========================================
elif menu == "🏛️ Perancangan Sistem (DFD & ERD)":
    st.title("🏛️ Perancangan Sistem SentiGlove")
    st.write(
        "Dokumentasi arsitektur sistem, aliran data (DFD), basis data relasional (ERD), dan landasan teori GloVe sesuai perancangan."
    )

    t_dfd, t_erd, t_theory = st.tabs(["DFD Level 0 & 1", "Skema ERD Basis Data", "Landasan Teori GloVe"])

    with t_dfd:
        st.subheader("DFD Level 0 (Diagram Konteks)")
        st.markdown("""
        Satu proses utama **0 (Sistem Analisis Sentimen Ulasan)** berinteraksi dengan dua entitas eksternal:
        - **Penjual:** Mengirimkan ulasan baru dan menerima label sentimen ulasan.
        - **Admin Data:** Memasok korpus data ulasan dan menerima laporan performa model.
        """)

        st.subheader("DFD Level 1 (Dekomposisi Proses)")
        st.markdown("""
        Sistem dibagi menjadi 4 proses utama:
        1. **1.0 Praproses Ulasan:** Case folding, normalisasi singkatan/slang, tokenisasi, dan penghapusan stopword (mempertahankan kata sentimen & negasi).
        2. **2.0 Matriks Kookurensi:** Menghitung kemunculan bersama pasangan kata dalam jendela konteks (window 5, bobot $1/\\text{jarak}$) -> disimpan ke `D2 Matriks X`.
        3. **3.0 Latih GloVe:** Meminimalkan loss kuadrat terbobot $J = \\sum f(X_{ij}) \\cdot (w_i \\cdot \\tilde{w}_j + b_i + \\tilde{b}_j - \\log X_{ij})^2$ dengan SGD/AdaGrad -> menghasilkan `D3 Vektor GloVe`.
        4. **4.0 Klasifikasi Sentimen:** Menghitung vektor rata-rata ulasan baru, lalu mengukur Cosine Similarity terhadap pusat kelas $c^+$ dan $c^-$ -> disimpan ke `D4 Hasil Sentimen`.
        """)

    with t_erd:
        st.subheader("Skema Basis Data Relasional (ERD SentiGlove)")
        st.markdown("""
        Terdapat 6 entitas inti relasional dalam basis data:
        1. **PENJUAL** (`id_penjual` PK, `nama_toko`)
           - Relasi 1-N ke PRODUK.
        2. **PRODUK** (`id_produk` PK, `id_penjual` FK, `nama_produk`, `kategori`)
           - Relasi 1-N ke ULASAN.
        3. **ULASAN** (`id_ulasan` PK, `id_produk` FK, `teks`, `rating_asli`, `label_asli`, `created_at`)
           - Relasi 1-1 ke HASIL_SENTIMEN.
        4. **MODEL_GLOVE** (`id_model` PK, `dimensi`, `x_max`, `alpha`, `loss_akhir`, `tanggal_latih`)
           - Relasi 1-N ke VEKTOR_KATA, KOOKURENSI, dan HASIL_SENTIMEN.
        5. **VEKTOR_KATA** (`id` PK, `id_model` FK, `kata`, `vektor_w`, `bias_b`)
        6. **KOOKURENSI** (`id` PK, `id_model` FK, `kata_i`, `kata_j`, `jumlah`)
        7. **HASIL_SENTIMEN** (`id_hasil` PK, `id_ulasan` FK, `id_model` FK, `skor_pos`, `skor_neg`, `label_prediksi`)
        """)

    with t_theory:
        st.subheader("2. Mengapa Memilih GloVe (Global Vectors for Word Representation)?")
        st.markdown("""
        - **Word2Vec (Skip-gram / CBOW):** Hanya mengandalkan jendela konteks lokal kata per kata secara berulang tanpa memanfaatkan statistik korpus global secara menyeluruh.
        - **LSA (Latent Semantic Analysis):** Memanfaatkan statistik global matriks dekomposisi SVD, namun sensitif terhadap kata yang sangat sering muncul dan tidak memetakan proporsi rasio probabilitas makna secara optimal.
        - **GloVe (Pennington dkk., 2014):** Menggabungkan keunggulan keduanya. Menggunakan matriks kookurensi statistik global korpus, lalu memfaktorkan rasio probabilitas kookurensi $P(k|i) / P(k|j)$ ke dalam ruang vektor log-bilinear. Akibatnya, kata dengan sentimen serupa menempati daerah berdekatan di ruang vektor 50-dimensi.
        """)

