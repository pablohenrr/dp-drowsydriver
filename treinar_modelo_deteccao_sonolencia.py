import os
import shutil
import yaml
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm

load_dotenv()

STORAGE_ACCOUNT_NAME = os.getenv('STORAGE_ACCOUNT_NAME')
STORAGE_ACCOUNT_KEY = os.getenv('STORAGE_ACCOUNT_KEY')
CONTAINER_NAME = os.getenv('CONTAINER_NAME')
DATASET_BLOB_PREFIX = ""  
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATASET_DIR = ROOT_DIR / "drowsy-driver-dataset"  
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
YOLO_MODEL = "yolov5/models/yolov5s.pt" 
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
VAL_SIZE = 0.2  

def _criar_blob_service_client():
    return BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=STORAGE_ACCOUNT_KEY
    )

def _baixar_dataset():
    blob_service_client = _criar_blob_service_client()
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    blobs = container_client.list_blobs(prefix=DATASET_BLOB_PREFIX)
    for blob in tqdm(blobs, desc="Baixando arquivos do Azure"):
        blob_path = blob.name

        relative_path = os.path.relpath(blob_path, DATASET_BLOB_PREFIX) 

        if relative_path.startswith("images/") and relative_path.endswith((".png", ".jpg", ".jpeg")):
            download_path = IMAGES_DIR / os.path.basename(relative_path)
        elif relative_path.startswith("labels/") and relative_path.endswith(".txt"):
            download_path = LABELS_DIR / os.path.basename(relative_path)
        else:
            print(f"Ignorando arquivo: {blob_path}")
            continue  

        try:
            with open(download_path, "wb") as download_file:
                download_file.write(container_client.download_blob(blob).readall())
                print(f"Baixado: {blob_path} para {download_path}") 
        except Exception as e:
            print(f"Erro ao baixar {blob.name}: {e}") 

def _dividir_dataset():
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    train_images, val_images = train_test_split(all_images, test_size=VAL_SIZE, random_state=42)

    for dataset_type in ["train", "val"]:
        for folder in ["images", "labels"]:
            os.makedirs(DATASET_DIR / dataset_type / folder, exist_ok=True)

    for image in train_images:
        shutil.copy(IMAGES_DIR / image, DATASET_DIR / "train" / "images" / image)
        label_file = image.replace(Path(image).suffix, ".txt")
        shutil.copy(LABELS_DIR / label_file, DATASET_DIR / "train" / "labels" / label_file)

    for image in val_images:
        shutil.copy(IMAGES_DIR / image, DATASET_DIR / "val" / "images" / image)
        label_file = image.replace(Path(image).suffix, ".txt")
        shutil.copy(LABELS_DIR / label_file, DATASET_DIR / "val" / "labels" / label_file)

def _criar_arquivos_yaml():
    data_yaml = {
        'train': str(DATASET_DIR / 'train/images'),
        'val': str(DATASET_DIR / 'val/images'),
        'nc': 2,
        'names': ['cansado', 'nao_cansado']
    }

    with open('yolov5/data/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, indent=2)

def _treinar_modelo():
    os.system(
        f"python yolov5/train.py --img {IMG_SIZE} --batch {BATCH_SIZE} "
        f"--epochs {EPOCHS} --data yolov5/data/data.yaml "
        f"--weights {YOLO_MODEL} --project yolov5/runs/train"
    )

def treinar_modelo_deteccao_sonolencia():
    _baixar_dataset() 
    _dividir_dataset()
    _criar_arquivos_yaml()
    _treinar_modelo()

if __name__ == "__main__":
    treinar_modelo_deteccao_sonolencia()