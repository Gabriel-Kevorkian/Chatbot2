import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import faiss
import pickle
import re
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from db import fetch_all_products
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
META_FILE = "product_metadata.pkl"
EMBEDDINGS_FILE = "embeddings.npy"


class EnhancedProductSearch:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initialize the search system with configurable model"""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.products = None
        self._load_or_build_index()

    def build_enhanced_sentence(self, product: Dict) -> str:
        """Create a rich text representation optimized for search"""
        # Core product info
        text_parts = [
            f"{product['brand']} {product['product_name']}",
            f"for {product['gender']}",
            f"size {product['size']}",
            f"{product['color']} color",
            f"${product['price']}",
            f"category {product['category']}",
        ]

        # Add description with some preprocessing
        desc = product.get('description', '').strip()
        if desc:
            # Clean up description - remove excessive whitespace, normalize
            desc = ' '.join(desc.split())
            text_parts.append(f"description: {desc}")

        # Add searchable keywords based on category
        keywords = self._generate_keywords(product)
        if keywords:
            text_parts.append(f"keywords: {' '.join(keywords)}")

        return ". ".join(text_parts)

    def _generate_keywords(self, product: Dict) -> List[str]:
        """Generate additional searchable keywords based on product attributes"""
        keywords = []

        # Category-specific keywords
        category = product.get('category', '').lower()
        category_keywords = {
            'shirts': ['top', 'blouse', 'tee', 'button-up', 'dress shirt'],
            'pants': ['trousers', 'jeans', 'slacks', 'bottoms'],
            'shoes': ['footwear', 'sneakers', 'boots', 'heels', 'flats'],
            'dresses': ['gown', 'frock', 'outfit'],
            'accessories': ['jewelry', 'bags', 'belts', 'watches'],
        }

        for cat, kwords in category_keywords.items():
            if cat in category:
                keywords.extend(kwords)

        # Color variations
        color = product.get('color', '').lower()
        color_variants = {
            'black': ['dark', 'ebony', 'midnight'],
            'white': ['ivory', 'cream', 'off-white'],
            'blue': ['navy', 'azure', 'cobalt'],
            'red': ['crimson', 'burgundy', 'scarlet'],
            'green': ['olive', 'forest', 'emerald'],
        }

        for base_color, variants in color_variants.items():
            if base_color in color:
                keywords.extend(variants)

        return keywords

    def _preprocess_query(self, query: str) -> str:
        """Enhance query with synonyms and corrections"""
        query = query.lower().strip()

        # Common search term expansions
        expansions = {
            'cheap': 'affordable low price budget',
            'expensive': 'premium luxury high-end',
            'shirt': 'shirt top blouse',
            'pants': 'pants trousers jeans',
            'shoe': 'shoe footwear sneaker',
            'dress': 'dress gown outfit',
        }

        for term, expansion in expansions.items():
            if term in query:
                query = query.replace(term, expansion)

        return query

    def build_faiss_index(self, force_rebuild: bool = False):
        """Build FAISS index with enhanced features"""
        if not force_rebuild and all(os.path.exists(f) for f in [INDEX_FILE, META_FILE, EMBEDDINGS_FILE]):
            logger.info("Index files already exist. Use force_rebuild=True to rebuild.")
            return

        logger.info("Fetching products from database...")
        products = fetch_all_products()

        logger.info("Generating enhanced text representations...")
        texts = [self.build_enhanced_sentence(p) for p in products]

        logger.info("Computing embeddings...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32  # Optimize batch size for memory
        )

        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Create FAISS index
        dimension = embeddings.shape[1]

        # Use IndexHNSWFlat for better performance on larger datasets
        if len(products) > 10000:
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is M parameter
            index.hnsw.efConstruction = 200
        else:
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        index.add(embeddings)

        # Save everything
        faiss.write_index(index, INDEX_FILE)
        np.save(EMBEDDINGS_FILE, embeddings)
        with open(META_FILE, "wb") as f:
            pickle.dump(products, f)

        logger.info(f"🔄 Successfully indexed {len(products)} products into FAISS.")

    def _load_or_build_index(self):
        """Load existing index or build new one"""
        try:
            if all(os.path.exists(f) for f in [INDEX_FILE, META_FILE]):
                self.index = faiss.read_index(INDEX_FILE)
                with open(META_FILE, "rb") as f:
                    self.products = pickle.load(f)
                logger.info(f"Loaded existing index with {len(self.products)} products")
            else:
                logger.info("Building new index...")
                self.build_faiss_index()
                self._load_or_build_index()  # Recursive call to load after building
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Rebuilding index...")
            self.build_faiss_index(force_rebuild=True)
            self._load_or_build_index()

    def semantic_search(
            self,
            query: str,
            top_k: int = 10,
            threshold: float = 0.5,
            filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Enhanced semantic search with filtering

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Dict of filters like {'category': 'shirts', 'gender': 'women'}
        """
        if not self.index or not self.products:
            raise RuntimeError("Search index not loaded")

        # Preprocess query
        enhanced_query = self._preprocess_query(query)

        # Compute query embedding
        query_vec = self.model.encode([enhanced_query], convert_to_numpy=True)
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # Search with larger initial results for filtering
        search_k = min(top_k * 3, len(self.products))  # Search more to account for filtering
        distances, indices = self.index.search(query_vec, search_k)

        results = []
        for i, score in zip(indices[0], distances[0]):
            if i == -1:  # FAISS returns -1 for invalid indices
                continue

            similarity = float(score)  # For IndexFlatIP, this is already cosine similarity
            if similarity < threshold:
                continue

            product = self.products[i].copy()
            product["similarity"] = round(similarity, 3)
            product["search_query"] = query  # Track original query

            # Apply filters if provided
            if filters and not self._matches_filters(product, filters):
                continue

            results.append(product)

        # Sort by similarity and limit results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _matches_filters(self, product: Dict, filters: Dict) -> bool:
        """Check if product matches the given filters"""
        for key, value in filters.items():
            product_value = product.get(key, "").lower()
            filter_value = str(value).lower()

            # Handle range filters for price
            if key == 'price_range':
                try:
                    price = float(product.get('price', 0))
                    min_price, max_price = value
                    if not (min_price <= price <= max_price):
                        return False
                except (ValueError, TypeError):
                    return False

            # Handle list filters (e.g., multiple categories)
            elif isinstance(value, list):
                if not any(str(v).lower() in product_value for v in value):
                    return False

            # Handle exact match filters
            elif filter_value not in product_value:
                return False

        return True

    def multi_query_search(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Search using multiple queries and combine results"""
        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.semantic_search(query, top_k=top_k, threshold=0.3)
            for result in results:
                product_id = result.get('id') or str(result)  # Fallback if no ID
                if product_id not in seen_ids:
                    seen_ids.add(product_id)
                    all_results.append(result)

        # Re-sort by similarity
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def get_similar_products(self, product_id: int, top_k: int = 5) -> List[Dict]:
        """Find products similar to a given product"""
        # Find the product
        target_product = None
        for i, product in enumerate(self.products):
            if product.get('id') == product_id:
                target_product = product
                target_index = i
                break

        if not target_product:
            return []

        # Get embedding for target product
        if os.path.exists(EMBEDDINGS_FILE):
            embeddings = np.load(EMBEDDINGS_FILE)
            query_vec = embeddings[target_index:target_index + 1]
        else:
            # Fallback: recompute embedding
            text = self.build_enhanced_sentence(target_product)
            query_vec = self.model.encode([text], convert_to_numpy=True)
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # Search for similar products
        distances, indices = self.index.search(query_vec, top_k + 1)  # +1 to exclude self

        results = []
        for i, score in zip(indices[0], distances[0]):
            if i == target_index or i == -1:  # Skip the product itself
                continue

            product = self.products[i].copy()
            product["similarity"] = round(float(score), 3)
            results.append(product)

        return results[:top_k]


# Global instance for backward compatibility
search_engine = EnhancedProductSearch()


# Backward compatible functions
def build_faiss_index():
    """Backward compatible function"""
    search_engine.build_faiss_index(force_rebuild=True)


def semantic_search(query: str, top_k: int = 3, threshold: float = 0.4) -> List[Dict]:
    """Backward compatible function"""
    return search_engine.semantic_search(query, top_k, threshold)


# Example usage functions
def search_with_filters(query: str, **filters) -> List[Dict]:
    """
    Search with filters
    Example: search_with_filters("red dress", category="dresses", price_range=(20, 100))
    """
    return search_engine.semantic_search(query, filters=filters)


def find_similar(product_id: int, count: int = 5) -> List[Dict]:
    """Find products similar to given product ID"""
    return search_engine.get_similar_products(product_id, count)


def multi_search(queries: List[str]) -> List[Dict]:
    """Search using multiple related queries"""
    return search_engine.multi_query_search(queries)


