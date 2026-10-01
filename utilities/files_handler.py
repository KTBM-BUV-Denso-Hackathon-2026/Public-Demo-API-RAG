import aiohttp

async def download_file(url, destination):
    """
    Downloads a file from the given URL and saves it to the specified destination.

    Args:
        url (str): The URL of the file to download.
        destination (str): The local path where the file should be saved.

    Returns:
        str: The path to the downloaded file.
    """ 
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(destination, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                return destination
            else:
                raise Exception(f"Failed to download file from {url}. Status code: {response.status}")

def open_file(file_path):
    """
    Opens a file and returns its content.

    Args:
        file_path (str): The path to the file to open.

    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return content

async def delete_file(file_path):
    """
    Deletes the specified file from the local filesystem.

    Args:
        file_path (str): The path to the file to delete.   

    Returns:
        bool: True if the file was deleted successfully, False otherwise.
    """
    import os

    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
        return False
    