# Detección de Noticias Falsas mediante Machine Learning

Proyecto educativo de clasificación de noticias en inglés utilizando **procesamiento de lenguaje natural (NLP)** y **regresión logística en Python**.

El notebook desarrolla el proceso completo: carga de datos, limpieza del texto, vectorización, entrenamiento y evaluación del modelo.

## Tecnologías utilizadas

- **Python**
- **pandas:** carga y manipulación de datos.
- **BeautifulSoup:** eliminación de etiquetas HTML.
- **NLTK:** tokenización, eliminación de palabras frecuentes y stemming.
- **scikit-learn:** vectorización, clasificación y evaluación.
- **Jupyter Notebook:** desarrollo y ejecución del proyecto.

## Conjunto de datos

Se utiliza el [Fake and Real News Dataset de Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), compuesto por **44.898 noticias**:

- **21.417** etiquetadas como reales (`REAL`).
- **23.481** etiquetadas como falsas (`FAKE`).

Los archivos incluyen título, contenido, categoría y fecha. El modelo utiliza el **contenido de la noticia**, almacenado en la columna `text`.

## Metodología

1. **Carga y etiquetado:** lectura de `True.csv` y `Fake.csv` y unión en
