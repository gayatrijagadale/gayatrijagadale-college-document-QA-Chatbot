"""Run this after putting PDFs in data/raw_pdfs/:   python build_index.py"""
from src.system import build_all

if __name__ == "__main__":
    build_all()
    print("Done. Next: python app.py   (chatbot)   or   python -m src.evaluate")
