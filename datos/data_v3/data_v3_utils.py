'''
******************************************************************************
------------ Útiles
******************************************************************************
'''

import glob
import xml.etree.ElementTree as ET
from PIL import Image
import time
import os

import cv2



path_anotaciones = 'anotaciones/'
path_imagenes = 'images/'
size_to_convert = 80, 80

path_piscina = 'clases/piscina'
path_parking = 'clases/rotonda'
path_rotonda = 'clases/rotonda'

ochenta_path = '80x80/'

# convierte en imagenes las clases
def crop_class(filename, x, y, w, h,clase):
    fichero = Image.open(path_imagenes + filename)
    crop_rectangle = (x, y, w, h)
    image = fichero.crop(crop_rectangle)
    #image = image.resize(size, Image.NEAREST)

    if clase == 'piscina':
        image.save('clases/piscina/'+ clase + '_'  +  time.time().__str__() + '.jpg')
    elif clase == 'rotonda':
        image.save('clases/rotonda/' + clase + '_' + time.time().__str__() + '.jpg')
    elif clase == 'parking':
        image.save('clases/parking/' + clase + '_' + time.time().__str__() + '.jpg')



#Transforma para cada fichero las coordenadas/bboxes de clases en imagenes JPG
def xml_to_images():

    i = 0
    print('INICIADO')
    start = time.time()
    for xml_file in glob.glob(path_anotaciones + '*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            filename = root.find('filename').text
            clase = member[0].text
            x = int(member[4][0].text)
            y = int(member[4][1].text)
            w = int(member[4][2].text)
            h = int(member[4][3].text)
            crop_class(filename,x,y,w,h,clase)
            i +=1
            if (i % 100) == 0:
                print('Imágenes procesadas: ',i)
    end = time.time()

    print('PROCESO FINALIZADO')
    print('TIEMPO TOTAL: ', end - start)

def resize_small_to_large(path, file):
    original_image = cv2.imread(os.path.join(path, file),3)
    print(file)
    resized_image = cv2.resize(original_image, size_to_convert,
                               interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(os.path.join(path, file), resized_image)


def normalizar_imag():

    # recorre todas las carpetas buscando imágenes
    for root, dirs, files in os.walk(ochenta_path):
        for file in files:
            if file.endswith(".jpg"):
                resize_small_to_large(root, file)


normalizar_imag()