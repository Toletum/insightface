import logging
import os
from sklearn.cluster import DBSCAN
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

output_dir = "./agrupadas"
os.makedirs(output_dir, exist_ok=True)

data = np.load("embeddings_paths.npz", allow_pickle=True)
embeddings = data["embeddings"]
paths = data["paths"]

logging.info("Clustering...")
labels = DBSCAN(eps=0.5, metric='cosine', min_samples=1).fit_predict(embeddings)
n_clusters = len(set(labels))
logging.info(f"{n_clusters} clusters encontrados.")

logging.info("Create symlink for visual testing...")
for label, path in zip(labels, paths):
    person_dir = os.path.join(output_dir, f"persona_{label}")
    os.makedirs(person_dir, exist_ok=True)
    link_name = os.path.join(person_dir, os.path.basename(path))
    if not os.path.exists(link_name):
        os.symlink(os.path.abspath(path), link_name)


logging.info("Saving...")
np.savez("labels.npz", embeddings=np.array(embeddings), paths=np.array(paths), labels=np.array(labels))
