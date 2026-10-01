from database.config import get_db
import psycopg2

def save_to_raw_data(columns, values):
    """
    Save data to the rag_rawdata table in the database.

    Args:
        columns (list): List of column names.
        values (list): List of values corresponding to the columns.
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create a string of placeholders for the values
            placeholders = ', '.join(['%s'] * len(values))
            # Create a string of column names
            columns_str = ', '.join(columns)
            # Create the SQL query
            query = f"INSERT INTO rag_rawdata ({columns_str}) VALUES ({placeholders})"
            # Execute the query with the provided values
            cursor.execute(query, values)
            conn.commit()

def save_to_rag_data(columns, values):
    """
    Save data to the rag_ragdata table in the database.

    Args:
        columns (list): List of column names.
        values (list): List of values corresponding to the columns.
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create a string of placeholders for the values
            placeholders = ', '.join(['%s'] * len(values))
            # Create a string of column names
            columns_str = ', '.join(columns)
            # Create the SQL query
            query = f"INSERT INTO rag_ragdata ({columns_str}) VALUES ({placeholders})"
            # Execute the query with the provided values
            cursor.execute(query, values)
            conn.commit()

def save_to_embedding_data(values):
    """
    Save data to the rag_embedding table in the database.

    Args:
        columns (list): List of column names.
        values (list): List of values corresponding to the columns.
    """
    with get_db() as conn:
        with conn.cursor() as cursor:
            # Create the SQL query
            query = f"INSERT INTO rag_chunkdata (id,chunk_content,chunk_index,metadata,vector,fts) VALUES (%s, %s, %s, %s, %s, to_tsvector('simple',%s))"
            
            # Execute the query with the provided values
            cursor.execute(query, values)
            conn.commit()