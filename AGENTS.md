# insightface

Scripts de reconocimiento facial: generan embeddings de caras con InsightFace, los agrupan por similitud y permiten buscar a qué grupo pertenece una imagen nueva.

## Visibilidad y rama
- Público.
- Rama por defecto: no verificado (no se detectó `.git` accesible).

## Stack
- Python.
- InsightFace 0.7.3 (modelo `buffalo_l`) con ONNX Runtime GPU (`onnxruntime-gpu`) y PyTorch 2.7.1 (CUDA 12.6).
- OpenCV (`opencv-python` y `-headless`), NumPy, scikit-learn (DBSCAN y `cosine_similarity`), tqdm.
- Dependencias fijadas en `requirements.txt` (72 líneas, incluyen paquetes `nvidia-*`; pensado para Linux con GPU NVIDIA).

## Estructura
- `training.py`: busca caras en `input_dir` y guarda `embeddings_paths.npz` (solo imágenes con una única cara).
- `clustering.py`: DBSCAN con distancia coseno (`eps=0.5`) sobre los embeddings; crea `agrupadas/persona_<n>/` con symlinks y guarda `labels.npz`.
- `model.py`: detecta la cara de `tofind.jpg`, busca la similitud coseno máxima con `labels.npz` (umbral 0.3) y abre el grupo con `viewnior`.
- `requirements.txt`: dependencias fijadas.
- `README.md`: orden de ejecución (no tiene título).

## Comandos
Orden de ejecución (según README):
1. `python training.py`
2. `python clustering.py`
3. `python model.py`

Instalar: `pip install -r requirements.txt` (no verificado: no se ejecutó).

## Configuración
- `training.py`: `input_dir = "/./Images"` (hardcodeado; la carpeta no está en el repo).
- `model.py`: `IMG = "tofind.jpg"` (hardcodeado) y umbral `0.3`.
- `clustering.py`: `eps=0.5`, `min_samples=1` y salida `./agrupadas` (hardcodeados).
- Ninguna variable de entorno.
- Los ficheros de entrada/salida (`embeddings_paths.npz`, `labels.npz`) se generan en el directorio de trabajo.

## Convenciones para agentes
- Ejecutar los scripts en orden: `model.py` necesita `labels.npz` y `clustering.py` necesita `embeddings_paths.npz`.
- Todos los modelos usan `providers=['CUDAExecutionProvider']` y `ctx_id=0`: requieren GPU NVIDIA con CUDA.
- `model.py` usa `viewnior` (visor de imágenes de Linux): no es portable.

## Estado
- No hay tests.
- GPU obligatoria: no hay fallback a CPU.
- `model.py` llama a `exit(1)` / `exit(0)` (builtin de `site`); conviene `sys.exit`.
- `model.py` abre el visor con `subprocess.run(..., check=True)`: si `viewnior` no está instalado, falla sin mensaje claro.
- `training.py` ignora silenciosamente las imágenes con 0 o más de 1 cara (solo registra "Archivo inválido" si no se lee la imagen).
- README con título vacío (`# `).
- Rutas hardcodeadas: si se cambian las carpetas, cambiar los scripts.
