import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential, load_model

from PIL import Image
import numpy as np
import time

import cv2

#general parametters
ciudadespath = 'pruebas/ciudades/ciudad6.jpg'
size = 48, 48
rango = 50

# probabilidades minimas de cada clase para ser mostrada
prob_minima_piscina = 0.4
prob_minima_rotonda = 0.9

#Si las otras clases con menor probabilidad superan estos valores, no se considerará un output valido
prob_comp_piscina = 0.01
prob_comp_rotonda = 0.0000001

# Regiones a comprobar (los outputs suelen ser muy grandes)
numShowRects = 5000

# load the class_indices saved in the earlier step
class_dictionary = np.load('class_indices.npy').item()

# fine tuning model
model_fine = load_model('fine_tuning.h5')

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
# ******************************Funciones AUXILIARES*************************************
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

# Comprueba si hay boxes con coordenadas muy parecidas
def is_similar(x, y, h ,w, candidates):

    lo_es = False

    for item in candidates:
        xi, yi, hi, wi = item[1]
        if (abs(x-xi) < rango and abs(y-yi) < rango and abs(h-hi) < rango and abs(w-wi) < rango):
            lo_es = True
            break
    return lo_es

# Comprueba si las probabilidades dadas se ajustan a los mínimos requeridos
def is_Valid(prob, label):

    valido = False

    valores = 0
    # Para cada clase comprueba si las probabilidades de las otras clases son lo suficientemente pequeñas
    for ele in prob:
        if label == 'piscina' and max(prob)>prob_minima_piscina:
            if ele < prob_comp_piscina:
                valores += 1
        if label == 'rotonda' and max(prob)>prob_minima_rotonda:
            if ele < prob_comp_rotonda:
                valores += 1

    if valores == 2:
        valido = True
    if label == 'parking':
        valido = False

    return valido
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Función que emplea la técnica de Fine Tuning
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
def predictMultiple_fine(image_sat, x, y, w, h):

    crop_rectangle = (x, y, w + x , h + y)
    image = image_sat.crop(crop_rectangle)
    image = image.resize(size)
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    probabilities = model_fine.predict(image)
    class_predicted = probabilities.argmax(axis=-1)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    # print('Classes: ', inv_map)

    label = inv_map[inID]

    if is_Valid(probabilities[0],label):
        # get the prediction label
        print("Image ID: {}, Label: {}".format(inID, label))

        print("Label", label, "probabilities", probabilities[0],"-coordenadas: ",crop_rectangle)

    return label, probabilities[0]

def selec_cv2():
    
    # imagen que emplearán los clasificadores
    im_clasiffier = Image.open(ciudadespath)
    
    
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    # read image
    im = cv2.imread('pruebas/ciudades/ciudad6.jpg')
    b, g, r = cv2.split(im)
    im = cv2.merge([r,g,b])


    ''''# resize image
    newHeight = 800
    newWidth = int(im.shape[1] * 800 / im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))'''

    # create Selective Search Segmentation Object using default parameters
    start_s = time.time()
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    end_s = time.time()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    #if (sys.argv[2] == 'f'):
    #ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method'''
    #elif (sys.argv[2] == 'q'):
    ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()

    candidates = set()

    j = 0
    start = time.time()

    # itereate over all the region proposals
    for i, rect in enumerate(rects):

        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect

            # si hay coordenadas de un box similares a otro que ya exista, se elimina
            if is_similar(x, y, w, h, candidates) == True:
                continue
            # calling the clasiffier
            label, prob = predictMultiple_fine(im_clasiffier, x, y, w, h)

            # si cumple con el mínimo de prob, se añade a candidatos
            if is_Valid(prob, label):
                candidates.add((label,(x, y, w, h)))
            j += 1
            if (j % 100 == 0):
                print("Regiones revisadas: ", j)
        else:
            break
    end = time.time()

    for item in candidates:
        print(item)

    print("tiempo procesamiento selective search: ", end_s - start_s)
    print("tiempo procesamiento: ", end - start)

    print('Regiones propuestas: ', len(rects))
    print("Regiones válidas: ", j)
    print("Regiones clasificadas: ", candidates.__len__())

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(im)
    for item in candidates:
        # print(x, y, w, h)
        x, y, w, h = item[1]
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        #añadir anotación al box
        ax.annotate(item[0], (x + w /2, y + h/2), color='w', weight='bold',
                    fontsize=6, ha='center', va='center')

    plt.show()

if __name__ == "__main__":
    selec_cv2()