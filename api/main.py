"""
Aplikasi Utama FastAPI SentiGlove
Menyediakan REST API untuk backend analisis sentimen ulasan produk berbasis GloVe.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.db import Base, engine
from api.routes import router

# Pastikan semua tabel basis data terinisialisasi
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SentiGlove API - Analisis Sentimen Ulasan E-Commerce",
    description=(
        "Backend REST API untuk aplikasi SentiGlove berbasis Word Embedding GloVe 50D, "
        "SQLAlchemy ORM, dan klasifikasi Centroid Cosine Similarity."
    ),
    version="1.0.0"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan Router
app.include_router(router)


@app.get("/")
def read_root():
    return {
        "app": "SentiGlove API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "description": "REST API Word Embedding GloVe untuk Analisis Sentimen Ulasan Produk E-commerce"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

