<div align="justify">

## Licencia

[![Licencia: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

Este proyecto fue desarrollado por *Superjosh769* con fines educativos y de impacto social.  
Está licenciado bajo *Creative Commons BY-NC 4.0*, lo que significa que:

- Puedes usarlo, modificarlo y compartirlo.
- *Debes dar crédito* al autor original.
- *No puedes usarlo con fines comerciales* ni lucrar con él de ninguna forma.

Más información: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)


Nosotros somos SeñalAIze, un equipo de estudiantes que ha desarrollado una Inteligencia Artifical que se encarga de la detección y traducción de la Lengua de Señas Mexicana (LSM) a través de una cámara.

Los pasos para crear tu red neuronal y que permita identificar tus propias señas son los siguientes:

*NOTA*
Lee los "PUNTOS IMPORTANTES A CONSIDERAR* al final de este documento para poder realizar el código de manera correcta.

1. Descarga el código que te proporcionamos llamado "Data.py". Este te permitirá crear carpetas con la cantidad de imágenes que desees de la seña de tu preferencia. Al ejecutar el código, mostrarás tu mano, y la computadora empezará tomar las fotos que utilizaremos para el entrenamiento de la red neuronal (mientras más imágenes, más tiempo, pero mucho mejor resultado).
2. Descarga y bre el archivo "SeguimientoManos.py" con la aplicación Pycharm (ver el final de este escrito para evitar descargas erróneas), que permitirá la correcta identificación y seguimiento de nuestras manos.
3. Tras haber obtenido todas las imágenes de todas las señas que deseemos, procederemos a crear una nueva carpeta titulada "data". Dentro de esta crearemos otras dos carpetas, tituladas "images" y "labels". Dentro de cada una de estas dos carpetas, crearemos otras dos, tituladas "train" y "val". En la carpeta "train" de "images", agregaremos todas las imágenes que tomamos de las señas; en la carpeta "val" de "images", agregaremos alrededor del 20% de las imágenes de cada seña (por ejemplo, 20 imágenes de la letra A, 20 de la B, y así sucesivamente).
4. Después de esto, buscaremos en nuestro navegador la página "makesense.ai", la cual nos permitirá etiquetar cada una de las imágenes que tomamos. Nos pedirá subir una carpeta, y nosotros le proporcionaremos la carpeta "train" de "images". Nos pedirá agregar el nombre de las etiquetas, y debemos de poner el mismo nombre con el que tomamos las fotos (por ejemplo, si las fotos tienen el nombre de "LetraA", la etiqueta también debe nombrarse "LetraA"). Tras terminar de etiquetar todas las imágenes, daremos click en la parte que dice "Actions",y presionaremos "export annotation". Aquí aparecerán varias opciones, y daremos click en la que dice "exportar zip en Formato YOLO". Estas etiquetas se descargarán, y las copiaremos y pegaremos en la carpeta "train" de "labels".
5. Repetimos el proceso, subiendo la carpeta "val" de "images" y copiando y pegando los resultados en la carpeta "val" de "labels".
6. La carpeta "data" la subiremos a nuestro google drive, acordándonos bien de la dirección que tiene la carpeta y no confundirnos.
7. Posteriormente, abriremos el archivo "custom.yaml". Aquí modificaremos tanto la cantidad como el nombre de las clases. Empieza del 0 en adelante, y el nombre de cada clase debe corresponder al de las etiquetas. En la parte superior, modificarás la sección "path" con la dirección de tu carpeta "data" de tu google drive. Finalmente, solo  guardas los cambios. Este archivo también lo subirás a tu google drive.
8. Luego, irás a la página google.colab, donde tendrás un tipo de "bloc de notas" pero de comandos. Aquí copiarás los códigos que se muestran en la imagen "google.colab" que te proporcionamos, modificando según tu dirección de google drive.
9. Posteriormente, solo tendrás que ir ejecutando los comandos uno por uno, hasta completar el entrenamiento.
10. Luego, el archivo titulado "best.pt" se copiará en tu google drive. Este archivo lo descargarás y lo guardarás en una carpeta nueva que contenga todos los pasos anteriores, para que no se pierda y todo tu proyecto quede en un solo lugar (aquí no importa el nombre que le pongas).
11. Finalmente abrirás el archivo llamado "inferencia.py", que también guardarás en la carpeta de tu proyecto. Ahora solo tendrás que ejecutar y ¡Listo! Tienes tu propio sistema de IA que detecta y traduce Lenguaje de Señas.


*PUNTOS IMPORTANTES A CONSIDERAR*
- La aplicación donde ejecutamos el código se llama Pycharm community edition 2022.3.1 (esta versión es gratuita).
- El programa te pedirá instalar algunas librerías, las cuales se encuentran en la imagen "librerías" y que puedes buscar cómo descargar en internet para evitar complicaciones o dudas.

</div>
