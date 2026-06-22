import psycopg2

def drop_everything():
    print("Dropping all tables...")
    try:
        conn = psycopg2.connect("dbname=employee_knowledge_db user=postgres password=Ham33ton! host=localhost")
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA public CASCADE;")
            cur.execute("CREATE SCHEMA public;")
        print("Database wiped successfully.")
    except Exception as e:
        print(f"Error wiping database: {e}")

if __name__ == "__main__":
    drop_everything()
