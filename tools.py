from langchain_core.tools import tool
from db import get_db_connection
from typing import Optional, List, Dict, Any

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


@tool
def query_products(
    category: Optional[str] = None,
    brand: Optional[str] = None,
    gender: Optional[str] = None,
    size: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    in_stock: bool = True,
    limit: int = 3
) -> List[Dict[str, Any]]:
    """
     Use this tool to search for products from our internal shop database only.

    You must call this tool whenever the user is asking about specific products, such as:
    - Finding products by category (e.g., shoes, laptops, shirts)
    - Filtering by brand (e.g., Nike, Adidas, Apple)
    - Searching within a price range (e.g., 100 to 300 dollars)
    - Selecting specific colors (e.g., black, red, blue)
    - Sorting by price or rating (ascending or descending)

    Parameters:
    - category: Product category (e.g. "laptop", "sneakers", "smartphone").
    - brand: Optional brand filter (e.g. "Nike", "Samsung").
    - min_price: Minimum price range (e.g. 100).
    - max_price: Maximum price range (e.g. 500).
    - color: Optional color filter (e.g. "black").
    - sort_by: Optional sorting preference: "price_asc", "price_desc", "rating", etc.

    This tool queries a MySQL database and returns matching products with name, price, and description.
    DO NOT make up product data — always use this tool.
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
        if min_price is not None:
            filters.append("p.price >= %s")
            params.append(min_price)
        if max_price is not None:
            filters.append("p.price <= %s")
            params.append(max_price)
        if in_stock:
            filters.append("p.in_stock = TRUE")

        filter_query = " AND ".join(filters) if filters else "1"

        sql = f"""
            SELECT p.name, p.description, p.price, p.size, p.color, p.gender, p.in_stock, p.images
            FROM products p
            JOIN categories c ON p.category_id = c.id
            JOIN brands b ON p.brand_id = b.id
            WHERE {filter_query}
            LIMIT %s
        """
        params.append(limit)

        cursor.execute(sql, params)
        results = cursor.fetchall()

        # Convert comma-separated images string to list
        for product in results:
            images_str = product.get('images', '')
            product['images'] = [img.strip() for img in images_str.split(',')] if images_str else []

        return results

    except Exception as e:
        return [{"error": str(e)}]

    finally:
        cursor.close()
        conn.close()


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

