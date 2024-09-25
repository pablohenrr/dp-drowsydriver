import cv2
import torch
from pathlib import Path

MODELO_DIR = Path("yolov5/runs/train/exp3")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODELO_DIR / 'weights/best.pt'), force_reload=True)

model.nc = 2
model.names = ['cansado', 'nao_cansado']

model.eval()

video_path = "ex01.mp4"
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise IOError("Erro ao abrir o vídeo.")

cv2.namedWindow('Classificação do Motorista', cv2.WINDOW_NORMAL)

colors = {
    'cansado': (0, 0, 255),    
    'nao_cansado': (0, 255, 0) 
}

while True:
    ret, frame = video.read()

    if not ret:
        break

    with torch.no_grad():
        results = model(frame)

    predictions = results.xyxy[0]

    for *box, conf, class_id in predictions:
        x1, y1, x2, y2 = map(int, box)
        classe = model.names[int(class_id)]
        label = f"{classe} {conf:.2f}"

        color = colors[classe]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        label_w, label_h = label_size
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + label_w, y1), color, -1)  

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Classificação do Motorista', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()