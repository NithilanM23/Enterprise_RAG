from services.database_service import _get_raw_connection
with _get_raw_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("UPDATE embeddings SET embedding_model = 'nomic-embed-text:latest' WHERE embedding_model IS NULL AND embedding IS NOT NULL;")
        print(f"Updated {cur.rowcount} rows")
