import cv2
import os
from ultralytics import YOLO

#Importar la clase
import SeguimientoManos as sm

#Lectura de la camara
cap = cv2.VideoCapture(1)



#Cambiar la resolución
cap.set(3, 1280)
cap.set(4, 720)

#Leer nuestro modelo
model = YOLO('best.pt')


#Declarar detector
detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    #Lectura de la videocaptura
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    #Extraer informacion de la mano
    frame = detector.encontrarmanos(frame, dibujar=False)

    #Posicion de una sola mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0, 255, 0])

    #Si hay mano
    if mano == 1:
        #Extraer la informacion del cuadro
        xmin, ymin, xmax, ymax = bbox

        #Asignamos margen
        xmin = xmin - 100
        ymin = ymin - 100
        xmax = xmax + 100
        ymax = ymax + 100

        #Realizar recorte de nuestra mano
        recorte = frame[ymin:ymax, xmin:xmax]

        #Redimensionamiento
        #recorte = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_CUBIC)

        #Extraer resultados
        resultados = model.predict(recorte, conf=0.70)

        #Si hay resulatdos
        if len(resultados) != 0:
            #Iteramos
            for results in resultados:
                masks = results.masks
                coordenadas = masks

                anotaciones = resultados[0].plot()

        cv2.imshow("RECORTE", anotaciones)

        #cv2.rectangle(frame, (xmin , ymin), (xmax, ymax), [0, 255, 0], 2)

    # Mostramos los Frames
    cv2.imshow("Vocales", frame)

    # Cerramos con lectura de teclado
    t = cv2.waitKey(1)
    # Salimos
    if t == 27:
        break


cv2.destroyAllWindows()
cap.release()


