"""
Skema Basis Data Relasional SentiGlove (Sesuai ERD Gambar 3 Proposal UTS NLP)
Mencakup 6 Entitas Inti + Entitas Kookurensi:
1. PENJUAL
2. PRODUK
3. ULASAN
4. MODEL_GLOVE
5. VEKTOR_KATA
6. KOOKURENSI
7. HASIL_SENTIMEN
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database.db import Base


def get_utc_now():
    return datetime.now(timezone.utc)


class Penjual(Base):
    """
    Entitas PENJUAL (id_penjual, nama_toko)
    Hubungan: 1 Penjual memiliki N Produk.
    """
    __tablename__ = "penjual"

    id_penjual = Column(String(50), primary_key=True, index=True)
    nama_toko = Column(String(150), nullable=False, index=True)

    produks = relationship("Produk", back_populates="penjual", cascade="all, delete-orphan")


class Produk(Base):
    """
    Entitas PRODUK (id_produk, id_penjual, nama_produk, kategori)
    Hubungan: N Produk dimiliki 1 Penjual; 1 Produk memiliki N Ulasan.
    """
    __tablename__ = "produk"

    id_produk = Column(String(50), primary_key=True, index=True)
    id_penjual = Column(String(50), ForeignKey("penjual.id_penjual"), nullable=False, index=True)
    nama_produk = Column(String(250), nullable=False, index=True)
    kategori = Column(String(100), nullable=False, index=True)

    penjual = relationship("Penjual", back_populates="produks")
    ulasans = relationship("Ulasan", back_populates="produk", cascade="all, delete-orphan")


class Ulasan(Base):
    """
    Entitas ULASAN (id_ulasan, id_produk, teks, rating_asli, label_asli, created_at)
    Hubungan: N Ulasan untuk 1 Produk; 1 Ulasan memiliki 1 HasilSentimen.
    """
    __tablename__ = "ulasan"

    id_ulasan = Column(Integer, primary_key=True, autoincrement=True, index=True)
    id_produk = Column(String(50), ForeignKey("produk.id_produk"), nullable=False, index=True)
    teks = Column(Text, nullable=False)
    rating_asli = Column(Integer, nullable=True)
    label_asli = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=get_utc_now)

    produk = relationship("Produk", back_populates="ulasans")
    hasil_sentimen = relationship("HasilSentimen", back_populates="ulasan", uselist=False, cascade="all, delete-orphan")


class ModelGlove(Base):
    """
    Entitas MODEL_GLOVE (id_model, dimensi, x_max, alpha, loss_akhir, tanggal_latih, keterangan)
    Mencatat versi model word embedding yang digunakan.
    """
    __tablename__ = "model_glove"

    id_model = Column(String(50), primary_key=True, index=True)
    dimensi = Column(Integer, default=50, nullable=False)
    x_max = Column(Float, default=100.0, nullable=False)
    alpha = Column(Float, default=0.75, nullable=False)
    loss_akhir = Column(Float, nullable=True)
    tanggal_latih = Column(DateTime, default=get_utc_now)
    keterangan = Column(String(250), nullable=True)

    vektor_katas = relationship("VektorKata", back_populates="model", cascade="all, delete-orphan")
    kookurensis = relationship("Kookurensi", back_populates="model", cascade="all, delete-orphan")
    hasil_sentimens = relationship("HasilSentimen", back_populates="model")


class VektorKata(Base):
    """
    Entitas VEKTOR_KATA (id_model, kata, vektor_w, bias_b)
    Menyimpan parameter representasi vektor kata dan bias hasil pelatihan GloVe.
    """
    __tablename__ = "vektor_kata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_model = Column(String(50), ForeignKey("model_glove.id_model"), nullable=False, index=True)
    kata = Column(String(100), nullable=False, index=True)
    vektor_w = Column(Text, nullable=False)  # JSON string array vektor kata
    bias_b = Column(Float, default=0.0)

    model = relationship("ModelGlove", back_populates="vektor_katas")


class Kookurensi(Base):
    """
    Entitas KOOKURENSI (id_model, kata_i, kata_j, jumlah X(i, j))
    Menyimpan nilai matriks kookurensi X per pasangan kata dalam korpus.
    """
    __tablename__ = "kookurensi"

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_model = Column(String(50), ForeignKey("model_glove.id_model"), nullable=False, index=True)
    kata_i = Column(String(100), nullable=False, index=True)
    kata_j = Column(String(100), nullable=False, index=True)
    jumlah = Column(Float, nullable=False)

    model = relationship("ModelGlove", back_populates="kookurensis")


class HasilSentimen(Base):
    """
    Entitas HASIL_SENTIMEN (id_hasil, id_ulasan, id_model, skor_pos, skor_neg, label_prediksi, created_at)
    Hubungan: 1 Ulasan menghasilkan 1 HasilSentimen; dapat ditelusuri ke ModelGlove.
    """
    __tablename__ = "hasil_sentimen"

    id_hasil = Column(Integer, primary_key=True, autoincrement=True, index=True)
    id_ulasan = Column(Integer, ForeignKey("ulasan.id_ulasan"), nullable=False, unique=True, index=True)
    id_model = Column(String(50), ForeignKey("model_glove.id_model"), nullable=False, index=True)
    skor_pos = Column(Float, nullable=False)
    skor_neg = Column(Float, nullable=False)
    label_prediksi = Column(String(20), nullable=False, index=True)
    created_at = Column(DateTime, default=get_utc_now)

    ulasan = relationship("Ulasan", back_populates="hasil_sentimen")
    model = relationship("ModelGlove", back_populates="hasil_sentimens")
