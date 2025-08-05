import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress warnings and info messages (0=all,1=info,2=warning,3=error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import faiss
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from db import fetch_all_products
import numpy as np

INDEX_FILE = "index.faiss"
META_FILE = "product_metadata.pkl"

# # Load the embedding model once
model = SentenceTransformer("all-mpnet-base-v2")

def build_sentence(product: Dict) -> str:
    return (
        f"{product['brand']} {product['product_name']} for {product['gender']} - "
        f"Size {product['size']}, Color: {product['color']}. "
        f"Price: ${product['price']}. Category: {product['category']}. "
        f"Description: {product['description']}"
    )

def build_faiss_index():
    products = fetch_all_products()
    texts = [build_sentence(p) for p in products]
    embeddings = model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(products, f)

    print(f"🔄 Indexed {len(products)} products into FAISS.")

def semantic_search(query: str, top_k: int = 3, threshold: float = 0.8) -> List[Dict]:
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        build_faiss_index()

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        products = pickle.load(f)

    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, dist in zip(indices[0], distances[0]):
        similarity = 1 - (dist / 2)
        if similarity >= threshold:
            product = products[i]
            product["similarity"] = round(similarity, 3)
            results.append(product)

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results


