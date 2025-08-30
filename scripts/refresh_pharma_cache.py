# scripts/refresh_pharma_cache.py
import time, sqlite3, os
from pharma.resolver import get_price_bd, resolve_bd_medicine, DB_PATH

def top_brands(limit=50):
    if not os.path.exists(DB_PATH): return []
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT brand FROM brands ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [r[0] for r in rows]

def main():
    for b in top_brands(50):
        try:
            resolve_bd_medicine(b)
            get_price_bd(b)
            time.sleep(1.2)  # be polite to the source site
        except Exception as e:
            print("Skip", b, e)

if __name__ == "__main__":
    main()
