from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

def upload_folder_to_blob(local_folder_path, blob_prefix=None):

    load_dotenv()
    storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
    storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
    container_name = os.getenv('CONTAINER_NAME')

    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=storage_account_key
    )
    container_client = blob_service_client.get_container_client(container_name)

    if blob_prefix is None:
        blob_prefix = os.path.basename(local_folder_path)

    subpastas_permitidas = ["images", "labels"] 

    for root, _, files in os.walk(local_folder_path):
        pasta_atual = os.path.basename(root) 
        if pasta_atual in subpastas_permitidas:
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                blob_name = os.path.join(blob_prefix, relative_path).replace("\\", "/")

                blob_client = container_client.get_blob_client(blob_name)
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"Arquivo {local_file_path} enviado para o blob {blob_name}")


if __name__ == "__main__":
    local_folder_path = "drowsy-driver-dataset"
    upload_folder_to_blob(local_folder_path, blob_prefix="")