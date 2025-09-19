# scripts/ingest_anatomy_images.py
import logging
from pathlib import Path
from typing import List, Tuple
from utils.pdf_image_extractor import index_anatomy_pdf
from config.env_config import DATA_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def discover_anatomy_pdfs() -> List[Tuple[str, str]]:
    """
    Dynamically discover anatomy PDFs in data/pdfs/anatomy/ directory.
    Returns list of (path, title) tuples.
    """
    anatomy_configs = []
    
    # Define anatomy PDF search directory
    anatomy_dir = DATA_DIR / "pdfs" / "anatomy"
    
    if not anatomy_dir.exists():
        logger.warning(f"Anatomy directory not found: {anatomy_dir}")
        return anatomy_configs
    
    pdf_files = list(anatomy_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} anatomy PDFs in {anatomy_dir}")
    
    for pdf_path in pdf_files:
        # Generate title from filename, making it more readable
        title = _generate_title_from_filename(pdf_path.stem)
        anatomy_configs.append((str(pdf_path), title))
        logger.info(f"  - {pdf_path.name} -> title: {title}")
    
    if not anatomy_configs:
        logger.warning("No anatomy PDFs found")
    
    return anatomy_configs


def _generate_title_from_filename(filename: str) -> str:
    """
    Generate a readable title from PDF filename.
    """
    # Clean up common patterns in anatomy PDF names
    title = filename.replace("_", " ").replace("-", " ")
    
    # Handle specific patterns
    title_mappings = {
        "human anatomy": "Human Anatomy",
        "ross and wilson": "Ross & Wilson: Anatomy & Physiology",
        "color atlas": "Color Atlas of Anatomy",
        "rohen": "Rohen Color Atlas of Anatomy",
        "netter": "Netter's Atlas of Human Anatomy",
        "gray": "Gray's Anatomy",
        "sobotta": "Sobotta Atlas of Human Anatomy",
        "atlas": "Atlas of Human Anatomy",
        "physiology": "Anatomy & Physiology",
    }
    
    title_lower = title.lower()
    for pattern, replacement in title_mappings.items():
        if pattern in title_lower:
            return replacement
    
    # Default: title case
    return title.title()


if __name__ == "__main__":
    # Discover anatomy PDFs dynamically
    anatomy_pdfs = discover_anatomy_pdfs()
    
    if not anatomy_pdfs:
        logger.error("No anatomy PDFs found to process")
        exit(1)
    
    for path, title in anatomy_pdfs:
        try:
            logger.info(f"Processing anatomy PDF: {path}")
            index_anatomy_pdf(path, title, namespace="anatomy")
            logger.info(f"Successfully indexed: {title}")
        except Exception as e:
            logger.error(f"[ERROR] {path}: {e}")
