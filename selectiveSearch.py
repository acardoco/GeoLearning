# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.preprocessing.image import img_to_array, load_img 
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
import selectivesearch
import numpy as np

from PIL import Image
import time

#general parametters
ciudadespath = 'pruebas/ciudades/ciudad6.jpg'
top_model_weights_path = 'bottleneck_fc_model.h5' 
size = 48, 48
prob_minima = 0.95

# load the class_indices saved in the earlier step
model_clasificador = load_model('my_model_v4.h5')

# fine tuning model
model_fine = load_model('fine_tuning.h5')

# build the VGG16 network
model_vgg16 = applications.VGG16(include_top=False, weights='imagenet')

# load the class_indices saved in the earlier step
class_dictionary = np.load('class_indices.npy').item()

#carga la imagen para el algoritmo de selective search
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="uint8" )
    return data

def is_Valid(prob):

    valido = False

    valores = 0

    if max(prob) > prob_minima:
        for ele in prob:
            if ele < 0.00001:
                valores += 1

    if valores == 2:
        valido = True

    return valido


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Función que a partir de unas coordenadas emplea el clasificador que tiene como base VGG16
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
def predictMultiple_vgg16(image_sat, x, y, w, h):

    num_classes = len(class_dictionary)

    crop_rectangle = (x, y, w + x , h + y)
    image = image_sat.crop(crop_rectangle)
    image = image.resize(size)
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model_vgg16.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    # obtener resultados
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]


    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    if is_Valid(probabilities[0]):
        # get the prediction label
        print("Image ID: {}, Label: {}".format(inID, label))

        print("Label", label, "probabilities", probabilities[0],"-coordenadas: ",crop_rectangle)

    return label, probabilities[0]

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

    if is_Valid(probabilities[0]):
        # get the prediction label
        print("Image ID: {}, Label: {}".format(inID, label))

        print("Label", label, "probabilities", probabilities[0],"-coordenadas: ",crop_rectangle)

    return label, probabilities[0]

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#Función que emplea el clasificador con redes convolucionales puestas manualmente
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
def predictMultiple_clasificador(image_sat, x, y, w, h):

    crop_rectangle = (x, y, w + x , h + y)
    image = image_sat.crop(crop_rectangle)
    image = image.resize(size)
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)
    class_predicted = model_clasificador.predict_classes(image)
    probabilities = model_clasificador.predict_proba(image)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    # print('Classes: ', inv_map)

    label = inv_map[inID]

    if is_Valid(probabilities[0]):
        # get the prediction label
        print("Image ID: {}, Label: {}".format(inID, label))

        print("Label", label, "probabilities", probabilities[0],"-coordenadas: ",crop_rectangle)

    return label, probabilities[0]

def main():

   # loading astronaut image
    img = load_image(ciudadespath)

    #imagen que emplearán los clasificadores
    im_clasiffier = Image.open(ciudadespath)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=200, sigma=0.5, min_size=10)

    candidates = set()  

    i = 0
    start = time.time()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions
        if r['size'] < 60:
            continue
        # distorted rects
        x, y, w, h = r['rect']

        if w / h > 1.2 or h / w > 1.2:
            continue
        #calling the clasiffier
        label, prob = predictMultiple_fine(im_clasiffier,x, y, w, h)

        #si cumple con el mínimo de prob, se añade a candidatos
        if is_Valid(prob):
            candidates.add(r['rect'])
        i += 1
        if (i % 100 == 0):
            print ("Regiones revisadas: ", i)


    end = time.time()
    print("tiempo procesamiento: ", end - start) 

    

    print("Regiones seleccionadas:", i)
    print("Regiones clasificadas:",candidates.__len__())
    
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        # print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()