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

1. **Carga y etiquetado:** lectura de `True.csv` y `Fake.csv` y unión en un único conjunto de datos.
2. **Preprocesamiento:** eliminación de ciertos prefijos de fuentes, HTML, URLs y puntuación; conversión a minúsculas, tokenización, eliminación de stopwords en inglés y stemming.
3. **Vectorización:** transformación del texto en recuentos de palabras mediante `CountVectorizer` (*Bag of Words*).
4. **Entrenamiento:** ajuste de un modelo `LogisticRegression(max_iter=1000)`.
5. **Evaluación:** cálculo de la exactitud (*accuracy*) sobre noticias reservadas para pruebas.
6. **Predicción:** clasificación de nuevos textos y consulta de las probabilidades asignadas a cada clase.

El vocabulario de cada experimento se aprende únicamente con los datos de entrenamiento. Los textos de prueba se transforman utilizando ese mismo vocabulario.

## Resultados

Las salidas guardadas en el notebook muestran:

- **1.000 noticias de entrenamiento y 500 de prueba:** accuracy del **95,0 %**.
- **40.000 noticias de entrenamiento y 2.000 de prueba:** accuracy del **98,7 %**.

Ambos experimentos utilizan muestreo con `random_state=42`. Como sus conjuntos de prueba son distintos, la comparación es orientativa.

## Cómo ejecutar el proyecto

1. Descarga el notebook de este repositorio.
2. Descarga los archivos `True.csv` y `Fake.csv` desde Kaggle.
3. Coloca ambos archivos dentro de `datasets/Fake_Real_News_Dataset/`, en la carpeta donde se encuentra el notebook.
4. Instala las dependencias:

   ```bash
   pip install jupyter pandas beautifulsoup4 nltk scikit-learn
   ```

5. Inicia Jupyter:

   ```bash
   jupyter notebook
   ```

6. Abre `5_Regresion_Logistica_Deteccion_Noticias_Falsas.ipynb` y ejecuta las celdas en orden.

El notebook descarga los recursos de NLTK necesarios: `punkt`, `punkt_tab` y `stopwords`. La primera ejecución requiere conexión a internet.

Para probar otros textos, modifica la lista `noticias_nuevas` de la sección final y vuelve a ejecutar las celdas de predicción.

## Alcance y limitaciones

Este proyecto permite practicar clasificación de texto con aprendizaje supervisado. El modelo aprende patrones del conjunto de datos; **no verifica hechos ni consulta fuentes externas**.

Los resultados corresponden a las particiones utilizadas y no garantizan el mismo rendimiento con noticias de otras fuentes, temas o épocas. El preprocesamiento está diseñado para inglés.

## Posibles mejoras

- Incorporar precision, recall, F1 y una matriz de confusión.
- Comparar la representación actual con TF-IDF.
- Revisar duplicados antes de separar entrenamiento y prueba.
- Evaluar el modelo con noticias de fuentes externas.
- Integrar el preprocesamiento y el modelo en un pipeline.

## Contexto

Proyecto de aprendizaje desarrollado a partir de un ejercicio de curso sobre regresión logística y procesamiento de texto.
