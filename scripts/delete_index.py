"""
Delete a Pinecone index by name.

Usage:
  python scripts/delete_index.py arobot-medical-pdfs
"""
from __future__ import annotations
import sys
import os
from config.env_config import PINECONE_API_KEY

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/delete_index.py <index_name>")
        sys.exit(2)
    name = sys.argv[1]
    if not PINECONE_API_KEY:
        print("PINECONE_API_KEY not configured in environment.")
        sys.exit(1)
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        names = [i.name for i in pc.list_indexes()]
        if name not in names:
            print(f"Index '{name}' not found. Available: {', '.join(names) if names else '(none)'}")
            sys.exit(1)
        pc.delete_index(name)
        print(f"Deleted index '{name}'.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

