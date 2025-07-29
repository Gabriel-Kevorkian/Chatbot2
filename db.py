import mysql.connector

def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="llm_suggestor"
    )
    return connection
def get_all_categories_and_brands():
    print("Getting all categories and brands...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM categories;")
        categories = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT name FROM brands;")
        brands = [row[0] for row in cursor.fetchall()]

        conn.close()

        return categories, brands
    except Exception as e:
        print("Error fetching categories/brands:", e)
        return [], []

def fetch_all_products():
    query = """
    SELECT 
        p.id,
        p.name AS product_name,
        c.name AS category,
        b.name AS brand,
        p.description,
        p.size,
        p.color,
        p.gender,
        p.price
    FROM products p
    LEFT JOIN categories c ON p.category_id = c.id
    LEFT JOIN brands b ON p.brand_id = b.id
    WHERE p.deleted_at IS NULL AND p.in_stock = 1
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

