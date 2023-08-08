# Análisis y Propiedades de Imágenes Solares 

En este directorio nos enfocamos en el tratamiento, procesamiento y análisis de imágenes solares. Nuestro objetivo final es generar bases de datos detalladas, donde cada entrada está asociada con una estampa de tiempo y los siguientes parámetros calculados de las imágenes:

### Entropía
La entropía es un concepto que cuantifica el nivel de desorden o imprevisibilidad de un conjunto de datos. La entroía se utiliza frecuentemente en el conteto de fotografías para analizar la complejidad o la riqueza de la información contenida en ellas.  Esta se calcula como: $H = -\sum(p_k * \log_2(p_k))$ donde $p_k$ es la intensidad de color normalizada en el pixel $k$ en la imagen, la normalización se toma del valor máximo en el píxel. 

**Referencias:**
1. Shannon, C.E. (1948), A Mathematical Theory of Communication. Bell System Technical Journal, 27: 379-423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
2. Thomas M. Cover and Joy A. Thomas. 2006. Elements of Information Theory (Wiley Series in Telecommunications and Signal Processing). Wiley-Interscience, USA.


- Media
- Desviación Estándar
- Dimensión Fractal
- Asimetría
- Kurtosis
- Uniformidad
- Suavidad Relativa
- Contraste Tamura
- Direccionalidad Tamura: La que esta en el paper,  ecuación dos.

Estos parámetros se han seleccionado por su relevancia en la caracterización de la textura y las propiedades de las imágenes solares.

## Estructura de Archivos

El directorio contiene los siguientes archivos clave:

- **image_services.py:** Este script alberga las funcionalidades generales que se utilizan a lo largo del proyecto para los diferentes cálculos. Entre las funcionalidades se incluyen la carga de imágenes, la extracción del disco solar, y otras utilidades relacionadas con el manejo de imágenes.

- **solar_image_analysis.py:** Este archivo contiene las funciones específicas para calcular cada uno de los parámetros listados anteriormente. Cada función está diseñada para procesar una imagen y extraer un parámetro específico.

- **image_processing_pipeline.ipynb:** Este Jupyter Notebook es donde se ensamblan todas las funcionalidades. Aquí, se realiza el proceso completo desde la carga de las imágenes hasta la generación de las bases de datos con los parámetros calculados.

- **script_ovatala.py:** Este script tiene la funcionalidad para calcular todos los parametros anteriormente descritos, pasando un rango de fechas, el tipo de imagen y la frecuencia. Para esto debe tener en cuenta la hora de inicio en la fecha es unica para cada tipo de imagen y la frecuencia, a continuacion se muestran:
    **hmiigr**: 
      - Hora incio: 00:00:00
      - tipo de imagen: hmiigr
      - frecuencia: 90T
    **eit171**:
      - Hora incio: 01:00:00
      - tipo de imagen: eit171
      - frecuencia: 12H
    **eit195**:
      - Hora incio: 01:13:00
      - tipo de imagen: eit195
      - frecuencia: 12H
    **eit284**:
      - Hora incio: 01:06:00
      - tipo de imagen: eit284
      - frecuencia: 12H
    **eit304**:
      - Hora incio: 01:19:00
      - tipo de imagen: eit304
      - frecuencia: 12H

    
**Nota:** En el jupyter *image_processing_pipeline* se puede ver como se pasan dichos parametros