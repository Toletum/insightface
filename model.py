import numpy as np
import cv2
import insightface
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
data = np.load("labels.npz", allow_pickle=True)
embeddings = data["embeddings"]
paths = data["paths"]
labels = data["labels"]

# Cargar modelo
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0)

# Imagen nueva
IMG="tofind.jpg"
img = cv2.imread(IMG)
if img is None:
    print("Imagen inválida")
    exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = model.get(img)

if not faces:
    print("No se detectó ninguna cara")
    exit(1)

emb = faces[0].embedding.reshape(1, -1)
sims = cosine_similarity(emb, embeddings)[0]
idx = np.argmax(sims)

sim_max = sims[idx]
threshold = 0.3
if sim_max < threshold:
    print(f"No hay coincidencias fuertes (similitud máx = {sim_max:.2f})")
    exit(0)


print(f"Label asignada: persona_{labels[idx]}")

persona_label = labels[idx]
fotos = [path for path, label in zip(paths, labels) if label == persona_label]

print(f"Fotos de persona_{persona_label}:")
for foto in fotos:
    print(foto)

import subprocess

comando = ["viewnior", f"agrupadas/persona_{labels[idx]}"]
subprocess.run(comando, check=True, capture_output=False)
