import numpy as np
import logging
import os
import cv2
import insightface
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

input_dir = "/./Images"

model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0)

logging.info("Escaneando imágenes...")
filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

embeddings = []
paths = []

for path in tqdm(filenames, desc="Procesando imágenes", unit="images"):
    try:
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"Archivo inválido: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = model.get(img)
        if len(faces) == 1:
            embeddings.append(faces[0].embedding)
            paths.append(path)
    except Exception as ex:
        logging.error(f"Error procesando {path}: {ex}")

np.savez("embeddings_paths.npz", embeddings=embeddings, paths=paths)
logging.info("Embeddings guardados en embeddings_paths.npz")
