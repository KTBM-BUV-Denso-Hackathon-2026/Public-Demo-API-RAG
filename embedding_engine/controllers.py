import json
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from embedding_engine.config import get_embedding

def chunk_to_vector(chunk):
    """
    Chuyển đổi một chunk văn bản thành vector embedding.

    Args:
        chunk (str): Chunk văn bản cần chuyển đổi.

    Returns:
        list: Vector embedding của chunk.
    """
    return get_embedding(chunk)

def chunks_process(chunks):
    """
    Xử lý danh sách các chunk văn bản, chuyển đổi chúng thành vector embedding.

    Args:
        chunks (list): Danh sách các chunk văn bản.
    Returns:
        list: Danh sách các vector embedding tương ứng với các chunk.
    """
    for chunk in chunks:
        vector = chunk_to_vector(chunk['text'])
        