import cv2
import os

#Importar la clase
import SeguimientoManos as sm

#Creacion de la carpeta
nombre = 'Letra Y'
direccion = '/Users/joshblrt/Desktop/'
carpeta = direccion + '/' + nombre

#Si no esta creada la carpeta
if not os.path.exists(carpeta):
    print("CARPETA CREADA: ", carpeta)
    #Creamos la carpeta
    os.makedirs(carpeta)

#Lectura de la camara
cap = cv2.VideoCapture(1)

cap.set(3, 1280)
cap.set(4, 720)

#Declaramos contador
cont = 0

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
        xmin = xmin - 40
        ymin = ymin - 40
        xmax = xmax + 40
        ymax = ymax + 40

        #Realizar recorte de nuestra mano
        recorte = frame[ymin:ymax, xmin:xmax]

        #Redimensionamiento
        #recorte = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_CUBIC)

        #Almacenar nuestras imagenes
        cv2.imwrite(carpeta + "/Y_{}.jpg".format(cont), recorte)

        #Aimentamos contador
        cont = cont + 1


        cv2.imshow("RECORTE", recorte)

        #cv2.rectangle(frame, (xmin , ymin), (xmax, ymax), [0, 255, 0], 2)

    # Mostramos los Frames
    cv2.imshow("LENGUAJE VOCALES", frame)

    # Cerramos con lectura de teclado
    t = cv2.waitKey(1)
    # Salimos
    if t == 27 or cont == 200:
        break


cv2.destroyAllWindows()
cap.release()