import celery
import os
import dotenv

dotenv.load_dotenv()

broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
backend_url = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

message_broker = celery.Celery('rag_worker', broker=broker_url, backend=backend_url)