"""
Skrip Peluncur Aplikasi SentiGlove
Penggunaan:
    python run.py             # Menjalankan Streamlit Web Dashboard (Port 8501)
    python run.py --api       # Menjalankan FastAPI REST Backend (Port 8000)
    python run.py --all       # Menjalankan FastAPI dan Streamlit secara bersamaan
    python run.py --test      # Menjalankan seluruh pengujian unit test
    python run.py --init-db   # Menginisialisasi basis data SQLite dan melatih model
"""

import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Peluncur Aplikasi SentiGlove")
    parser.add_argument("--api", action="store_true", help="Jalankan FastAPI REST Server saja (Port 8000)")
    parser.add_argument("--all", action="store_true", help="Jalankan FastAPI dan Streamlit bersamaan")
    parser.add_argument("--test", action="store_true", help="Jalankan unit tests SentiGlove")
    parser.add_argument("--init-db", action="store_true", help="Inisialisasi & seeding basis data SQLite")
    args = parser.parse_args()

    if args.init_db:
        print(">>> Menjalankan inisialisasi basis data...")
        from database.init_db import init_database
        init_database()
        return

    if args.test:
        print(">>> Menjalankan Unit Tests SentiGlove Engine...")
        sys.exit(subprocess.call([sys.executable, "-m", "unittest", "tests/test_engine.py"]))

    if args.api:
        print(">>> Menjalankan FastAPI Backend pada http://127.0.0.1:8000 ...")
        print(">>> Dokumentasi Swagger API: http://127.0.0.1:8000/docs")
        sys.exit(subprocess.call([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]))

    if args.all:
        print(">>> Menjalankan FastAPI Backend dan Streamlit Frontend...")
        p_api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"])
        try:
            subprocess.call([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"])
        finally:
            p_api.terminate()
        return

    # Default: Streamlit UI
    print(">>> Menjalankan Streamlit Web Dashboard pada http://localhost:8501 ...")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", "frontend/app.py"]))


if __name__ == "__main__":
    main()

