from utilities.files_handler import open_file
import uuid

def content_extractor(file_path):
    """
    Extracts the content from a file.

    Args:
        file_path (str): The path to the file to extract content from.

    Returns:
        str: The extracted content of the file.
    """
    content = open_file(file_path)
    return content

def split_chunk_data(file_path, chunk_size=1000):
    """
    Splits the content of a file into chunks of specified size.

    Args:
        file_path (str): The path to the file to split.
        chunk_size (int): The size of each chunk in characters. Default is 1000.

    Returns:
        list: A list of strings, each representing a chunk of the file content.
    """
    content = content_extractor(file_path)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    chunks_data = []
    current_chunk = ""
    chunk_index = 0
    
    for p in paragraphs:
        # Gom các đoạn nhỏ liền kề
        if len(current_chunk) + len(p) < 1200:
            current_chunk += ("\n\n" + p) if current_chunk else p
        else:
            if current_chunk:
                chunks_data.append({
                    "id": str(uuid.uuid4()), # ID duy nhất cho vector DB
                    "text": current_chunk,
                    "metadata": {
                        "source": file_path,
                        "chunk_index": chunk_index,
                        "char_count": len(current_chunk)
                    }
                })
                chunk_index += 1
            current_chunk = p
            
    # Xử lý chunk cuối cùng
    if current_chunk:
        chunks_data.append({
            "id": str(uuid.uuid4()),
            "text": current_chunk,
            "metadata": {
                "source": file_path,
                "chunk_index": chunk_index,
                "char_count": len(current_chunk)
            }
        })
        
    return chunks_data