# 

## Crear Embeddings
Busca en las imágenes las caras y crea su embedding para el modelo

input_dir variable con el directorio con las imágenes

```bash
python training.py
```

Guarda el array de embeddings (embedding y ruta de la imagen)

## Clustering

Agrupa los embeddings (las caras) por similitud dando una label por cara igual, luego crea un directorio con esa
label y hace un symlink para validación visual.

```bash
python clustering.py
```

Guarda el array de labels (embedding, ruta de la imagen y label)

## Validar una imagen

Dada un imagen (IMG), busca la cara y si hay crea el embedding y busca ese embedding en la lista de embeddings 
creadas por training.py

Para validación abre la visor viewnior en la directorio de la label encontrada

```bash
python model.py
```
