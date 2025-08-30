# scripts/ingest_anatomy_images.py
from utils.pdf_image_extractor import index_anatomy_pdf

DOCS = [
    ("data/pdfs/anatomy/Human Anatomy.pdf", "Human Anatomy"),
    ("data/pdfs/anatomy/ross-and-wilson-anatomy-and-physiology-in-health-a.pdf", "Ross & Wilson: A&P"),
    ("data/pdfs/anatomy/color-atlas-of-anatomy-a-photog.-study-of-the-human-body-7th-ed.-j.-rohen-et-al.-lippincott-2011.pdf", "Rohen Color Atlas of Anatomy"),
]

if __name__ == "__main__":
    for path, title in DOCS:
        try:
            index_anatomy_pdf(path, title, namespace="anatomy")
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
