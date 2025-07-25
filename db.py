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

def wants_more_products(user_message: str) -> bool:
    triggers = ['more', 'other', 'another', 'else', 'different']
    return any(word in user_message.lower() for word in triggers)


