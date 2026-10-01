from fastapi import FastAPI
from broker_setup.config import message_broker
from celery import shared_task
from embedding_engine.controllers import chunk_to_vector
from utilities.files_handler import *
from utilities.data_preprocessor import content_extractor, split_chunk_data

from database.controllers import *

app = FastAPI()

@shared_task(name='process.rag.task', bind=True, ignore_result=True)
async def process_rag_task(task_id: int, file_url: str):
    """
    This task is responsible for sending RAG data to a separate microservice for processing.
    The processing will be asynchronous, so we do not need to receive the data immediately from the microservice.
    Instead, we will receive the data after the microservice has finished processing it.
    Therefore, this task will only send the RAG data to the microservice and does not need to receive the data back immediately.
    """
    # Download the file
    try:
        file_path = await download_file(file_url, destination=f'/tmp/{task_id}.md')

        content = content_extractor(file_path)
        save_to_raw_data("content", content)

        chunks = split_chunk_data(file_path)
        save_to_raw_data("chunk_data", chunks)

        for chunk in chunks:
            vector = chunk_to_vector(chunk['text'])
            value = (chunk['id'],chunk['text'],chunk['metadata'['chunk_index']], chunk['metadata'],vector,chunk['text'])
            save_to_embedding_data(values=value)


        # Delete the file after processing
        await delete_file(file_path)
        return 200
    except:
        return 500

    # Here you can implement the logic to send the RAG data to the FastAPI microservice for processing.
    # For example, you can use an HTTP client like `requests` or `httpx` to make a POST request to the FastAPI endpoint.
    # You can also log the task_id or any other relevant information for debugging purposes.

"""
if __name__ == '__main__':
    uvicorn.run(app)
"""