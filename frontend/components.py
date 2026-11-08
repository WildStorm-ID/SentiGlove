"""
Komponen UI, Visualisasi Grafik, dan Styling Streamlit untuk SentiGlove
"""

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st


def apply_custom_styles():
    """Menyematkan styling CSS custom untuk tampilan modern, elegan, dan rapi."""
    st.markdown("""
    <style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Badges */
    .badge-pos {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.85rem;
        border: 1px solid #059669;
    }
    .badge-neg {
        background-color: #881337;
        color: #fda4af;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.85rem;
        border: 1px solid #e11d48;
    }

    /* Step Box */
    .calc-step {
        background-color: #1e222d;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def sentiment_donut_chart(pos_count: int, neg_count: int) -> alt.Chart:
    """Membuat donut chart proporsi sentimen Positif vs Negatif."""
    total = pos_count + neg_count
    if total == 0:
        pos_count = 1
        neg_count = 1

    data = pd.DataFrame({
        "Sentimen": ["Positif", "Negatif"],
        "Jumlah": [pos_count, neg_count],
        "Persentase": [round(pos_count / (pos_count + neg_count) * 100, 1),
                       round(neg_count / (pos_count + neg_count) * 100, 1)]
    })

    chart = alt.Chart(data).mark_arc(innerRadius=65, stroke="#1e293b", strokeWidth=2).encode(
        theta=alt.Theta(field="Jumlah", type="quantitative"),
        color=alt.Color(
            field="Sentimen",
            type="nominal",
            scale=alt.Scale(
                domain=["Positif", "Negatif"],
                range=["#10b981", "#ef4444"]
            ),
            legend=alt.Legend(orient="bottom", title=None)
        ),
        tooltip=["Sentimen", "Jumlah", alt.Tooltip("Persentase:Q", format=".1f", title="Persentase (%)")]
    ).properties(
        width=280,
        height=280
    )
    return chart


def category_sentiment_bar_chart(df_category: pd.DataFrame) -> alt.Chart:
    """Membuat bar chart perbandingan sentimen per kategori produk."""
    chart = alt.Chart(df_category).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("kategori:N", title="Kategori Produk", axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("jumlah:Q", title="Jumlah Ulasan"),
        color=alt.Color(
            "sentimen:N",
            scale=alt.Scale(domain=["Positif", "Negatif"], range=["#10b981", "#ef4444"]),
            legend=alt.Legend(orient="top", title="Sentimen")
        ),
        tooltip=["kategori", "sentimen", "jumlah"]
    ).properties(
        height=320
    )
    return chart


def vector_dimension_bar_chart(vector: list) -> alt.Chart:
    """Membuat bar chart visualisasi nilai vektor 50 dimensi."""
    df_vec = pd.DataFrame({
        "Dimensi": [f"D{i+1}" for i in range(len(vector))],
        "Nilai": vector,
        "Sign": ["Positif" if v >= 0 else "Negatif" for v in vector]
    })

    chart = alt.Chart(df_vec).mark_bar().encode(
        x=alt.X("Dimensi:N", sort=None, title="Indeks Dimensi (1 - 50)"),
        y=alt.Y("Nilai:Q", title="Nilai Koordinat"),
        color=alt.Color(
            "Sign:N",
            scale=alt.Scale(domain=["Positif", "Negatif"], range=["#38bdf8", "#f43f5e"]),
            legend=None
        ),
        tooltip=["Dimensi", alt.Tooltip("Nilai:Q", format=".4f")]
    ).properties(
        height=220
    )
    return chart

