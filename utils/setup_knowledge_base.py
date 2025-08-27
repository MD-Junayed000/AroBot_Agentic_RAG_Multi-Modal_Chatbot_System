"""
Setup script for knowledge base ingestion
"""
import os
import sys
import pandas as pd
from pathlib import Path
from core.vector_store import PineconeStore
from core.embeddings import Embedder
from .data_ingestion import load_pdfs
from config.env_config import (
    PINECONE_API_KEY, PINECONE_PDF_INDEX, PINECONE_MEDICINE_INDEX,
    DATA_DIR, WEB_SCRAPE_DIR, EMBEDDING_DIMENSION
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_pdf_knowledge_base():
    """Setup PDF knowledge base in Pinecone"""
    try:
        logger.info("Setting up PDF knowledge base...")
        
        # Initialize vector store for PDFs
        pdf_store = PineconeStore(
            index_name=PINECONE_PDF_INDEX,
            dimension=384
        )
        
        # Load PDF documents
        logger.info(f"Loading PDFs from {DATA_DIR}")
        docs = load_pdfs(str(DATA_DIR))
        
        if not docs:
            logger.warning("No PDF documents found!")
            return False
        
        logger.info(f"📄 Found {len(docs)} PDF pages to index")
        
        # Prepare texts and metadata
        texts = [doc["text"] for doc in docs]
        metadatas = [doc["meta"] for doc in docs]
        
        # Show chunking information
        total_chars = sum(len(text) for text in texts)
        avg_chunk_size = total_chars // len(texts) if texts else 0
        logger.info(f"📊 Chunking info:")
        logger.info(f"   • Total chunks: {len(texts)}")
        logger.info(f"   • Average chunk size: {avg_chunk_size} characters")
        logger.info(f"   • Total text: {total_chars:,} characters")
        logger.info(f"   • Estimated batches: {(len(texts) + 63) // 64}")
        
        # Upsert to Pinecone
        logger.info("🚀 Uploading to Pinecone...")
        pdf_store.upsert_texts(texts, metadatas)
        
        logger.info(f"Successfully indexed {len(docs)} PDF pages to {PINECONE_PDF_INDEX}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up PDF knowledge base: {e}")
        return False

def setup_medicine_knowledge_base():
    """Setup medicine CSV knowledge base in Pinecone"""
    try:
        logger.info("Setting up medicine knowledge base...")
        
        # Initialize vector store for medicine data
        medicine_store = PineconeStore(
            index_name=PINECONE_MEDICINE_INDEX,
            dimension=384
        )
        
        # Load medicine CSV files
        generic_path = WEB_SCRAPE_DIR / "generic.csv"
        medicine_path = WEB_SCRAPE_DIR / "medicine.csv"
        
        if not generic_path.exists() or not medicine_path.exists():
            logger.error("Medicine CSV files not found!")
            return False
        
        logger.info("Loading medicine CSV files...")
        generic_df = pd.read_csv(generic_path)
        medicine_df = pd.read_csv(medicine_path)
        
        # Process and merge data (similar to your notebook code)
        logger.info("Processing medicine data...")
        
        # Merge the dataframes
        merged_df = pd.merge(
            generic_df, 
            medicine_df, 
            left_on='generic name', 
            right_on='generic', 
            how='inner'
        )
        
        logger.info(f"Merged data contains {len(merged_df)} records")
        
        # Clean HTML tags (simplified version)
        def clean_html(text):
            if pd.isna(text):
                return ""
            import re
            clean = re.compile('<.*?>')
            return re.sub(clean, '', str(text))
        
        # Clean description columns
        description_columns = [
            'indication description', 'therapeutic class description',
            'pharmacology description', 'dosage description',
            'side effects description', 'contraindications description'
        ]
        
        for col in description_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].apply(clean_html)
        
        # Create comprehensive medicine descriptions
        logger.info("Creating medicine descriptions...")
        
        def create_medicine_description(row):
            description_parts = []
            
            # Basic information
            if pd.notna(row.get('generic name')):
                description_parts.append(f"Generic Name: {row['generic name']}")
            
            if pd.notna(row.get('brand name')):
                description_parts.append(f"Brand Name: {row['brand name']}")
            
            if pd.notna(row.get('drug class')):
                description_parts.append(f"Drug Class: {row['drug class']}")
            
            if pd.notna(row.get('strength')):
                description_parts.append(f"Strength: {row['strength']}")
            
            if pd.notna(row.get('dosage form')):
                description_parts.append(f"Dosage Form: {row['dosage form']}")
            
            # Medical information
            for col in description_columns:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    clean_text = str(row[col]).strip()
                    if len(clean_text) > 10:  # Only include substantial content
                        description_parts.append(f"{col.title()}: {clean_text}")
            
            return " | ".join(description_parts)
        
        # Create descriptions
        merged_df['medicine_description'] = merged_df.apply(create_medicine_description, axis=1)
        
        # Prepare data for indexing
        texts = []
        metadatas = []
        
        for idx, row in merged_df.iterrows():
            description = row['medicine_description']
            if len(description) > 50:  # Only index substantial descriptions
                texts.append(description)
                metadatas.append({
                    'id': f"medicine_{idx}",
                    'generic_name': str(row.get('generic name', '')),
                    'brand_name': str(row.get('brand name', '')),
                    'drug_class': str(row.get('drug class', '')),
                    'source': 'medicine_database'
                })
        
        logger.info(f"💊 Prepared {len(texts)} medicine records for indexing")
        
        # Show chunking information
        total_chars = sum(len(text) for text in texts)
        avg_chunk_size = total_chars // len(texts) if texts else 0
        logger.info(f"📊 Medicine data info:")
        logger.info(f"   • Total records: {len(texts)}")
        logger.info(f"   • Average record size: {avg_chunk_size} characters")
        logger.info(f"   • Total text: {total_chars:,} characters")
        logger.info(f"   • Estimated batches: {(len(texts) + 63) // 64}")
        
        # Upsert to Pinecone
        logger.info("🚀 Uploading to Pinecone...")
        medicine_store.upsert_texts(texts, metadatas)
        
        logger.info(f"Successfully indexed {len(texts)} medicine records to {PINECONE_MEDICINE_INDEX}")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up medicine knowledge base: {e}")
        return False

def verify_knowledge_bases():
    """Verify that knowledge bases are set up correctly"""
    try:
        logger.info("Verifying knowledge bases...")
        
        # Test PDF knowledge base
        pdf_store = PineconeStore(
            index_name=PINECONE_PDF_INDEX,
            dimension=384
        )
        
        pdf_results = pdf_store.query("medical anatomy", top_k=3)
        logger.info(f"PDF KB test query returned {len(pdf_results)} results")
        
        # Test medicine knowledge base
        medicine_store = PineconeStore(
            index_name=PINECONE_MEDICINE_INDEX,
            dimension=384
        )
        
        medicine_results = medicine_store.query("diabetes medication", top_k=3)
        logger.info(f"Medicine KB test query returned {len(medicine_results)} results")
        
        return len(pdf_results) > 0 and len(medicine_results) > 0
        
    except Exception as e:
        logger.error(f"Error verifying knowledge bases: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting knowledge base setup...")
    
    # Check if API key is set
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY not found in environment!")
        return False
    
    success = True
    
    # Setup PDF knowledge base
    if not setup_pdf_knowledge_base():
        success = False
    
    # Setup medicine knowledge base
    if not setup_medicine_knowledge_base():
        success = False
    
    # Verify setup
    if success:
        if verify_knowledge_bases():
            logger.info("✅ Knowledge base setup completed successfully!")
        else:
            logger.warning("⚠️ Knowledge base setup completed but verification failed")
            success = False
    else:
        logger.error("❌ Knowledge base setup failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
