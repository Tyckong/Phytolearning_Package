# Phytolearning Package

**Phytolearning** es un paquete de código abierto en Python diseñado para automatizar la detección, segmentación morfológica y extracción de biométricas de complejos estomáticos en imágenes de microscopía.
Esta herramienta está diseñada para operar eficientemente en hardware estándar (CPUs) mediante una arquitectura en cascada que integra un modelo de detección de una etapa (**YOLO26s**) y un modelo fundacional destilado (**MobileSAM**).

## Características Principales
* **Detección Jerárquica:** Localiza simultáneamente el complejo estomático y el poro en su interior.
* **Segmentación Zero-Shot:** Aísla el área exacta del poro utilizando recortes matemáticos y un mecanismo de fallback automático ante oclusiones o bajo contraste.
* **Procesamiento Adaptativo:** Mejora de contraste basado en CLAHE enfocado en el canal de luminancia para facilitar el análisis de imágenes con ruido visual.
* **Diagnóstico Hídrico:** Clasifica el estado de la planta (*Healthy*, *Moderate Stress*, *Severe Drought*) calculando el porcentaje de apertura estomática.

---

## Instalación Local
Por el momento, el paquete se instala clonando o descargando este directorio de forma local. Se recomienda el uso de un entorno virtual como `venv` o `conda`.
Esto se puede realizar con el comando: 

   ``python -m venv <Nombre_del_entorno>``

Con el entorno creado, este se puede activar de distintas formas dependiendo del sistema operativo:

Windows: ``<Nombre_del_entorno>\Scripts\activate``

Mac/Linux: ``source <Nombre_del_entorno>/bin/activate`` 

Una vez dentro del entorno, no deberían haber problemas de dependencias, por lo que es seguro instalar el paquete.

1. Abre tu terminal y ve hasta la carpeta raíz del proyecto:
   cd <ruta_hacia>/phytolearning_pack

2. Instala el paquete y todas sus dependencias:
    pip install .

(El sistema descargará automáticamente librerías requeridas como OpenCV, Ultralytics, MobileSAM y timm.)

Una vez instalado, el comando phytolearning estará disponible globalmente en tu terminal.
El comando básico analiza todas las imágenes dentro de una carpeta y guarda las máscaras resultantes en un directorio de salida:

    phytolearning -i ./ruta_a_imagenes -o ./ruta_de_salida

El paquete viene con parámetros ajustables para los umbrales biológicos según la especie vegetal analizada:

    phytolearning -i ./imagenes -o ./resultados --aperture 10.0 --healthy 30.0 --moderate 70.0

## Argumentos de Consola:
-i, --input: Ruta al archivo de imagen o directorio de imágenes a analizar.

-o, --output: Directorio donde se guardarán las imágenes segmentadas (Default: ./phyto_output).

--aperture: Umbral mínimo de área (%) para considerar un poro biológicamente abierto (Default: 12.5).

--healthy: Umbral máximo de poros cerrados (%) para diagnosticar 'Healthy Plant' (Default: 25.0).

--moderate: Umbral máximo de poros cerrados (%) para diagnosticar 'Moderate Hydric Stress' (Default: 75.0).

## Uso como librería de Python:
Si se desea integrar la inferencia dentro de scripts propios de investigación, se puede importar la función central:
    from phytolearning import analyze_images
    
    analyze_images(input_path="./mis_muestras", output_dir="./mis_resultados")  #Ejecutar el pipeline con parámetros por defecto...
    analyze_images(                                                             #...O especificando umbrales personalizados.
        input_path="./mis_muestras/hoja1.jpg", 
        output_dir="./resultados_experimentales",
        aperture_th=15.0,
        healthy_th=20.0,
        moderate_th=60.0
    )
