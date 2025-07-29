# main.py
from db import fetch_all_products
from tools import embed_products

def main():
    print("🔄 Fetching product data from database...")
    products = fetch_all_products()
    print(f"✅ {len(products)} products loaded.")

    print("🧠 Generating sentence embeddings...")
    embedded_products = embed_products(products)

    # Just print the first one to verify
    print("\n🔍 Example:")
    print("Sentence:", embedded_products[0]["product_name"])
    print("Embedding (first 5 dims):", embedded_products[0]["embedding"][:5])

if __name__ == "__main__":
    main()
