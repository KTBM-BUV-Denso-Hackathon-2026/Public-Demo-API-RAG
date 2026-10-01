import json
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """
    Lấy embedding vector từ văn bản đầu vào.
    """
    return model.encode(text).tolist()