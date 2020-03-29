from math import sqrt
import math
import cv2
import numpy as np
import skimage.io
from skimage import data
from skimage.feature import blob_dog
from skimage.color import rgb2gray
from skimage import transform as tf
from skimage.transform import rotate
from skimage.feature import (match_descriptors, corner_harris,corner_peaks, ORB, plot_matches)
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch


grado=0
intervalo=5
E1=0
E2=0
n1=0
entradax=''
entraday=''
g=0
x1=0
y1=0

            #norte    #noreste  #este    sureste  sur   suroeste oeste noroeste
posiciones = []
filename = ''
dst=''
matches_array = []
img = ''
imagen = ''
imgEscalada=cv2.imread('flores.jpg')
escalas=[0.25,0.50,2,4]

def openFile():
    global filename,img
    filename = askopenfilename()
    img = cv2.imread(filename)

def defaultDoG():
    image = img
    image_gray = rgb2gray(image)
    blobs_dog = blob_dog(image_gray,max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    blobs = blobs_dog
    print (len(blobs))
    fig, ax = plt.subplots(1,figsize=(9, 3), sharex=True, sharey=True)
    ax.set_title('Default DoG')
    ax.imshow(image)
    
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
        ax.add_patch(c)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    
def escalar():
    
    global filename,img,matches_array,escalas,imagen
    matches_array=[]
    imagen=cv2.imread(filename)
    pixelesO = []
    pixelesT = []
    escalada1=0
    escalada2=0
    height, width = imagen.shape[:2]
    for i in escalas:
        try:
            res = cv2.resize(imagen,None,fx=i, fy=i, interpolation = cv2.INTER_CUBIC)
            keypointsO, keypointsT = detector(res)
            
            for i in range(len(keypointsO)):
                x, y, r= blobs
                escalada1=x*i
                escalada2=y*i
                pixelesO.append((escalada1,escalada2))
            
            print("escala: ",i)
            comparar2(pixelesT,pixelesO,keypointsO)
            
            pixelesO.clear()
            pixelesT.clear()
        except:
            print('No se detectó keypoints')  
def rotar():
    
    global filename,img,matches_array,imagen
    coincidencia = []
    originalesAmanita=[]
    transformadas=[]
    originales=[]
    grados=[]
    imagen=cv2.imread(filename)
    rows,cols = imagen.shape[:2]
    hipotenusa=pow(pow(rows,2)+pow(cols,2),1/2)/2
    img = cv2.copyMakeBorder(imagen, int(hipotenusa-(cols//2)), int(hipotenusa-(rows//2)),int(hipotenusa-(cols//2)), int(hipotenusa-(rows//2)), cv2.BORDER_CONSTANT, value=None)
    for i in range(0, 360, g):
        grados.append(i)
        dst = rotate(img, i)
        keypointsO,keypointsT = detector(dst)
        for i in range(len(keypointsO)):
            a,b=rotate_around_point_lowperf((keypointsO[i][0],keypointsO[i][1]),math.radians(g),(keypointsO[i][0]/2,keypointsO[i][1]/2))
            originalesAmanita.append((a,b))
        print("girando: ",i)
        comparar(keypointsT,keypointsO,originalesAmanita,imagen,dst)
        originalesAmanita.clear()
        
        
def rotate_around_point_lowperf(point, radians, origin):
    x, y = point
    ox, oy = origin
    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy
def transladar():
    global filename,matches_array,intervalo,x1,y1,posiciones,dst,imagen
    coincidencia=[]
    concatenarx=''
    concatenary=''
    originalesAmanita=[]
    coordenadasT=[]
    coordenadasO=[]
    imagen = cv2.imread(filename)
    
    
    
    img = cv2.copyMakeBorder(imagen, int(y1), int(y1),int(x1), int(x1), cv2.BORDER_CONSTANT, value=None)
    rows,cols = img.shape[:2]
    for xf,yf in posiciones:
        
        print(xf,yf)
        try:
            M = np.float32([[1,0,xf],[0,1,yf]])
            dst = cv2.warpAffine(img,M,(cols,rows))
            keypointsO,keypointsT = detector(dst)
            for i in range(len(keypointsO)):
                originalesAmanita.append((keypointsO[i][0]+xf,keypointsO[i][1]+yf))    

            comparar(keypointsT,keypointsO,originalesAmanita,imagen,dst)
            originalesAmanita.clear()
            
        except:
            print('No se detectó keypoints')   

    
def detector(dst):
    originales=[]
    transformadas=[]
    global filename
    
    image_gray = rgb2gray(dst)
    img_path_original = cv2.imread(filename)
    img_original = rgb2gray(img_path_original)

    blobs_dog = blob_dog(img_original, max_sigma=25, threshold=0.1,)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    blobsO = blobs_dog
    for blob in blobsO:
        y, x, r = blob
        originales.append((x,y))
  

    blobs_dog2 = blob_dog(image_gray, max_sigma=25, threshold=0.1,)
    blobs_dog2[:, 2] = blobs_dog2[:, 2] * sqrt(2)
    blobsT = blobs_dog2
    for blob in blobsT:
        y, x, r = blob
        transformadas.append((x,y))

    print("Total de keypoints img original: ",len(originales))
    print("Total de keypoints img transformada: ",len(transformadas))
    return  originales,transformadas
    image_gray=None


def graficar(coincidenciasO,coincidenciasT,imagen,dst,originales,transformadas):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.imshow(imagen)
    ax2.imshow(dst)

    coordsA = "data"
    coordsB = "data"
    for blob in coincidenciasT:
                x, y, = blob

                c = plt.Circle((x, y), 2.0, color='red', linewidth=2, fill=False)
                ax2.add_patch(c)
    for blob1 in coincidenciasO:
                x, y= blob1
                c = plt.Circle((x, y),2.0, color='red', linewidth=2, fill=False)
                ax1.add_patch(c)
    for i in range(len(originales)):
        con = ConnectionPatch(xyA=coincidenciasT[i], xyB=coincidenciasO[i], coordsA=coordsA, coordsB=coordsB,axesA=ax2, axesB=ax1,arrowstyle="-", shrinkB=5,color="yellow")
        ax2.add_artist(con)

    plt.show()
def comparar(keypointsT,keypointsO,originalesAmanita,imagen,dst):
    acertado=[]
    fallidos=[]
    coincidenciasO=[]
    coincidenciasT=[]
    try:
        for i in range(0,len(keypointsO)):
            result=pow(pow(keypointsT[i][0]-originalesAmanita[i][0],2)+pow(keypointsT[i][1]-originalesAmanita[i][1],2),1/2)
            if(result<=3 and result>=-3):
                acertado.append(result)
                coincidenciasO.append(keypointsO[i])
                coincidenciasT.append(keypointsT[i])
            else:
                fallidos.append(result)
        graficar(coincidenciasO,coincidenciasT,imagen,dst,keypointsO,keypointsT)
        print("acertados: ",acertado)
        print("fallos: ",fallidos)
    except:
        pass

def interfaz():
    ventana = tk.Tk()
    ventana.title("KeyPoints DoG")
    ventana.geometry('380x300')
    ventana.configure(background='dark turquoise')
    butonGrises = tk.Button(ventana,text="Default DoG",fg="blue",command=defaultDoG, width=50)
    butonGrises.pack(side=tk.TOP)
    butonMax = tk.Button(ventana,text="Escalar",fg="blue",command=escalar, width=50)
    butonMax.pack(side=tk.TOP)
    butonSombras = tk.Button(ventana,text="Rotar",fg="blue",command=interfazRotar, width=50)
    butonSombras.pack(side=tk.TOP)
    butonTrasladar = tk.Button(ventana,text="Trasladar",fg="blue",command=interfazTrasladar, width=50)
    butonTrasladar.pack(side=tk.TOP)
    buton = tk.Button(ventana,text="Abrir",fg="blue",command=openFile, width=50)
    buton.pack(side=tk.TOP)
    ventana.mainloop()   
def interfazRotar():
    global grado
    
    root = tk.Tk()
    root.geometry('380x300')
    root.title("Ventana")
    
    root.configure(background='dark turquoise')
    placeHolder=tk.Label(root, text="Ingresa los grados a rotar")
    placeHolder.pack()
    grado=tk.Entry(root,textvariable=n1)
    grado.pack()
    b = tk.Button(root, text="Rotar", command=funcionRotar,width=10)
    b.pack()
    root.mainloop()

def interfazTrasladar():
    global E1,E2
    root = tk.Tk()
    root.geometry('380x300')
    root.title("Ventana")
    
    root.configure(background='dark turquoise')
    
    L1 = tk.Label(root, text="Inserta X inicial")
    L1.pack()
    E1 = tk.Entry(root, bd =3,textvariable=entradax)
    E1.pack()
    L1 = tk.Label(root, text="Inserta Y inicial")
    L1.pack()
    E2 = tk.Entry(root, bd =3,textvariable=entraday)
    E2.pack()
    b = tk.Button(root, text="Trasladar", command=funcionTraslado,width=10)
    b.pack()
    root.mainloop()
def funcionTraslado():
    global x1,y1,posiciones
    
    x1 = int(E1.get())
    y1 =int(E2.get())
    posiciones = [(0,-y1), (-x1,-y1), (-x1,0), (-x1,y1), (0,y1),(x1,y1), (x1,0), (x1,-y1)]
    transladar()
def funcionRotar():
    global g
    g = int(grado.get())
    rotar()

interfaz()