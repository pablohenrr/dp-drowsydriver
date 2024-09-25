import os
import csv
import re

DATASET_DIR = "drowsy-driver-dataset"  
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

ARQUIVO_ANOTACOES = "drowsy-driver-dataset/rotulagem.csv"

def gerar_arquivos_yolo_txt(arquivo_anotacoes):

    os.makedirs(LABELS_DIR, exist_ok=True) 

    with open(arquivo_anotacoes, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 
        for linha in reader:

            match = re.search(r'/([^/]+)\.png$', linha[4])
            if match:
                nome_imagem = match.group(1) + ".png"
            else:
                print(f"Nome da imagem não encontrado na linha: {linha}")
                continue

            classe = 0 if linha[5] == "Cansado" else 1 

            nome_arquivo_txt = os.path.splitext(nome_imagem)[0] + ".txt"
            caminho_arquivo_txt = os.path.join(LABELS_DIR, nome_arquivo_txt)

            with open(caminho_arquivo_txt, 'w') as f:
                f.write(f"{classe} 0.5 0.5 1 1\n") 

if __name__ == "__main__":
    gerar_arquivos_yolo_txt(ARQUIVO_ANOTACOES)
    print("Arquivos .txt gerados com sucesso na pasta 'labels'!")