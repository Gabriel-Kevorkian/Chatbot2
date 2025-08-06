from langchain_core.tools import tool
from db import get_db_connection
from typing import Optional, List, Dict, Any
from semantic_search import semantic_search
import json
from decimal import Decimal

#
# @tool
# def list_products_by_category(category_name: str) -> str:
#     """List 2–3 in-stock products in a specific category."""
#     print("Listing products by category")
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT p.name, p.description, p.price
#             FROM products p
#             JOIN categories c ON p.category_id = c.id
#             WHERE c.name = %s AND p.in_stock = 1
#             LIMIT 3;
#         """, (category_name,))
#         results = cursor.fetchall()
#         conn.close()
#
#         if not results:
#             return f"No products found in category: {category_name}"
#
#         return "\n".join([f"{name} - {desc} (${price})" for name, desc, price in results])
#     except Exception as e:
#         return f"Error: {str(e)}"
#
#
# @tool
# def list_brands_by_category(category_name: str) -> str:
#     """List all brands that have products in a specific category."""
#     print("Listing brands by category")
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT DISTINCT b.name
#             FROM products p
#             JOIN categories c ON p.category_id = c.id
#             JOIN brands b ON p.brand_id = b.id
#             WHERE c.name = %s;
#         """, (category_name,))
#         results = cursor.fetchall()
#         conn.close()
#
#         if not results:
#             return f"No brands found in category: {category_name}"
#
#         return f"Brands in {category_name}: " + ", ".join([row[0] for row in results])
#     except Exception as e:
#         return f"Error: {str(e)}"
#
#
# @tool
# def search_products(gender: str, category: str, in_stock: bool = True) -> str:
#     """Search in-stock products by gender and category."""
#     print("Searching products by gender and category")
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT p.name, p.price
#             FROM products p
#             JOIN categories c ON p.category_id = c.id
#             WHERE p.gender = %s AND c.name = %s AND p.in_stock = %s;
#         """, (gender, category, in_stock))
#         results = cursor.fetchall()
#         conn.close()
#
#         if not results:
#             return f"No in-stock products for {gender} in category {category}"
#
#         return "\n".join([f"{name} - ${price}" for name, price in results])
#     except Exception as e:
#         return f"Error: {str(e)}"
#
#
# @tool
# def get_product_details(product_name: str) -> str:
#     """Get full details of a product by its name."""
#     print("Getting product details")
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT p.description, p.price, p.quantity, p.size, p.color, p.gender, p.in_stock
#             FROM products p
#             WHERE p.name = %s;
#         """, (product_name,))
#         result = cursor.fetchone()
#         conn.close()
#
#         if not result:
#             return f"Product '{product_name}' not found."
#
#         desc, price, qty, size, color, gender, in_stock = result
#         return (f"Description: {desc}\nPrice: ${price}\nQuantity: {qty}\nSize: {size}\n"
#                 f"Color: {color}\nGender: {gender}\nIn Stock: {'Yes' if in_stock else 'No'}")
#     except Exception as e:
#         return f"Error: {str(e)}"
#
#
# @tool
# def filter_products_by_price(max_price: float, category: str) -> str:
#     """List products under a specific price in a given category."""
#     print("Filtering products by price")
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT p.name, p.price
#             FROM products p
#             JOIN categories c ON p.category_id = c.id
#             WHERE p.price <= %s AND c.name = %s AND p.in_stock = 1;
#         """, (max_price, category))
#         results = cursor.fetchall()
#         conn.close()
#
#         if not results:
#             return f"No products found under ${max_price} in category {category}"
#
#         return "\n".join([f"{name} - ${price}" for name, price in results])
#     except Exception as e:
#         return f"Error: {str(e)}"
#
#
# @tool
# def search_by_size_and_category(size: str, category: str) -> str:
#     """Find in-stock products of a specific size in a category."""
#     print("Searching products by size")
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT p.name, p.price
#             FROM products p
#             JOIN categories c ON p.category_id = c.id
#             WHERE p.size = %s AND c.name = %s AND p.in_stock = 1;
#         """, (size, category))
#         results = cursor.fetchall()
#         conn.close()
#
#         if not results:
#             return f"No in-stock products of size {size} in category {category}"
#
#         return "\n".join([f"{name} - ${price}" for name, price in results])
#     except Exception as e:
#         return f"Error: {str(e)}"
#
#
# @tool
# def get_product_images(product_name: str) -> str:
#     """Retrieve all image links for a given product."""
#     print("Retrieving product images")
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("""
#             SELECT images
#             FROM products
#             WHERE name = %s;
#         """, (product_name,))
#         result = cursor.fetchone()
#         conn.close()
#
#         if not result:
#             return f"No product named '{product_name}' found."
#
#         # Assuming images is stored as comma-separated links
#         images_str = result[0]
#         if not images_str:
#             return f"No images found for '{product_name}'"
#
#         links = images_str.split(",")
#         return "\n".join(links)
#     except Exception as e:
#         return f"Error: {str(e)}"
#--------------------------------------------------------------------------------------------------------------------------------------
@tool
def query_products(
    category: Optional[str] = None,
    brand: Optional[str] = None,
    gender: Optional[str] = None,
    size: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    color: Optional[str] = None,         # Added color
    in_stock: bool = True,
    limit: int = 3,
    offset: int = 0,
    sort_by: Optional[str] = None        # Added sort_by
) -> List[Dict[str, Any]]:
    """
        Queries the internal products database to retrieve a list of products matching
        the given filters and preferences. This tool must be used to fetch any product
        information instead of generating or guessing product details.

        Parameters:
        - category (str, optional): Filter products by category name (e.g., "men-shoes", "laptops").
        - brand (str, optional): Filter products by brand name (e.g., "Nike", "Apple").
        - gender (str, optional): Filter products by gender specification (e.g., "men", "women").
        - size (str, optional): Filter products by size (e.g., "M", "10").
        - min_price (float, optional): Minimum price filter.
        - max_price (float, optional): Maximum price filter.
        - color (str, optional): Filter products by color (e.g., "black", "red").
        - in_stock (bool, default=True): Filter only products currently in stock.
        - limit (int, default=3): Maximum number of products to return.
        - offset (int, default=0): Number of products to skip, used for pagination.
        - sort_by (str, optional): Sort order of results, supported values include
          "price_asc", "price_desc", "name_asc", "name_desc".

        Returns:
        - List of dictionaries, each containing product details including name, description,
          price, size, color, gender, stock status, and a list of image URLs.

        Notes:
        - This tool queries the database directly and must be used for all product searches.
        - Pagination is supported via the `limit` and `offset` parameters.
        - Sorting is supported for price and name fields.
        - Image URLs are parsed into a list from a comma-separated string.
        """
    print("Querying products...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        filters = []
        params = []

        if category:
            filters.append("c.name = %s")
            params.append(category)
        if brand:
            filters.append("b.name = %s")
            params.append(brand)
        if gender:
            filters.append("p.gender = %s")
            params.append(gender)
        if size:
            filters.append("p.size = %s")
            params.append(size)
        if color:
            filters.append("p.color = %s")
            params.append(color)
        if min_price is not None:
            filters.append("p.price >= %s")
            params.append(min_price)
        if max_price is not None:
            filters.append("p.price <= %s")
            params.append(max_price)
        if in_stock:
            filters.append("p.in_stock = TRUE")

        filter_query = " AND ".join(filters) if filters else "1"

        # Handle sorting
        order_clause = ""
        if sort_by == "price_asc":
            order_clause = "ORDER BY p.price ASC"
        elif sort_by == "price_desc":
            order_clause = "ORDER BY p.price DESC"
        elif sort_by == "name_asc":
            order_clause = "ORDER BY p.name ASC"
        elif sort_by == "name_desc":
            order_clause = "ORDER BY p.name DESC"
        # Add more if needed

        sql = f"""
            SELECT p.name, p.description, p.price, p.size, p.color, p.gender, p.in_stock, p.images
            FROM products p
            JOIN categories c ON p.category_id = c.id
            JOIN brands b ON p.brand_id = b.id
            WHERE {filter_query}
            {order_clause}
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        cursor.execute(sql, params)
        results = cursor.fetchall()

        for product in results:
            # Fix Decimal serialization
            if isinstance(product.get('price'), Decimal):
                product['price'] = float(product['price'])

            # Fix image list
            images_raw = product.get('images', '')
            try:
                product['images'] = json.loads(images_raw)
            except Exception:
                product['images'] = []

        print("🔍 Tool Output (query_products):", results)
        if not results:
            if offset > 0:
                return json.dumps({
                    "status": "end_of_list",
                    "message": "That's all the products we have in this category.",
                    "results": []
                })
            else:
                return json.dumps({
                    "status": "empty",
                    "message": "No matching products found for your filters.",
                    "results": []
                })

        concise_results = []
        for p in results:
            concise_results.append({
                "name": p["name"],
                "price": float(p["price"]),
                "color": p.get("color", ""),
                "size": p.get("size", ""),
                "in_stock": p.get("in_stock", False),
                "image": p.get("images", [""])[0] if p.get("images") else ""
            })
        return json.dumps({
            "status": "success",
            "results": concise_results
        })




    except Exception as e:
        return [{"error": str(e)}]

    finally:
        try:
            cursor.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass



@tool
def list_brands_by_category(category_name: str) -> str:
    """List all brands that have products in a specific category."""
    print("Listing brands by category...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT b.name
            FROM products p
            JOIN categories c ON p.category_id = c.id
            JOIN brands b ON p.brand_id = b.id
            WHERE c.name = %s;
        """, (category_name,))
        results = cursor.fetchall()
        conn.close()

        if not results:
            return f"No brands found in category: {category_name}"

        brands_list = ", ".join(row[0] for row in results)
        return f"Brands in {category_name}: {brands_list}"

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_product_details(product_name: str, include_images: bool = False) -> str:
    """
    Get detailed info about a product by name.
    If include_images is True, append image URLs to the output.
    """
    print("Getting product details...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT p.name, p.description, p.price, p.size, p.color, p.gender, p.in_stock, p.images
            FROM products p
            WHERE p.name = %s
            LIMIT 1
        """, (product_name,))
        product = cursor.fetchone()
        conn.close()

        if not product:
            return f"No product named '{product_name}' found."

        response = (
            f"Product: {product['name']}\n"
            f"Description: {product['description']}\n"
            f"Price: ${product['price']}\n"
            f"Size: {product['size']}\n"
            f"Color: {product['color']}\n"
            f"Gender: {product['gender']}\n"
            f"In Stock: {'Yes' if product['in_stock'] else 'No'}"
        )

        if include_images:
            images_str = product.get('images', '')
            if images_str:
                images_list = [img.strip() for img in images_str.split(',')]
                images_text = "\n".join(images_list)
                response += f"\nImages:\n{images_text}"
            else:
                response += "\nNo images available."

        return response

    except Exception as e:
        return f"Error: {str(e)}"




@tool
def semantic_search_tool(query: str, top_k: int = 3) -> str:
    """Semantic search for products based on a free-text query."""
    results = semantic_search(query, top_k)

    if not results:
        return "I'm sorry, I couldn't find any matching products."

    output = "🧠 Based on your query, here are some products I found:\n\n"
    for product in results:
        output += (
            f"🛍️ {product['product_name']} ({product['brand']})\n"
            f"👕 Category: {product['category']}\n"
            f"💰 Price: ${product['price']}\n"
            f"📦 Size: {product['size']} | Color: {product['color']}\n"
            f"🧾 Description: {product['description']}\n\n"
        )

    return output.strip()



