# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
import cv2

from PIL import Image


ciudadespath = 'pruebas/ciudades/ciudad2.jpg'

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="uint8" )
    return data
'''
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#funcion para predecir varias imagenes a partir de un directorio dado
#--------------------------------------------------------------------
#--------------------------------------------------------------------
def predictMultiple(directory, imag_number, imag_format, imag_dim):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    for i in range(0,imag_number):

        # add the path to your test image below
        image_path = 'pathimagen' + i + '.' + imag_format

        orig = cv2.imread(image_path)

        print("[INFO] loading and preprocessing image...")
        image = load_img(image_path, target_size=(imag_dim, imag_dim))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255

        image = np.expand_dims(image, axis=0)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # get the bottleneck prediction from the pre-trained VGG16 model
        bottleneck_prediction = model.predict(image)

        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))

        model.load_weights(top_model_weights_path)

        # use the bottleneck prediction on the top model to get the final
        # classification
        class_predicted = model.predict_classes(bottleneck_prediction)

        probabilities = model.predict_proba(bottleneck_prediction)


def asigProb():
    # loading test image
    img = load_image(ciudadespath)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=100, sigma=0.9, min_size=10)

    candidates = set()    

    #a cada region se le aplica la red entrenada
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])
'''
def main():

   # loading astronaut image
    img = load_image(ciudadespath)

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=100, sigma=0.9, min_size=10)

    #-----ejemplo BORRAR LUEGO
    im = Image.open(ciudadespath)
    crop_rectangle = (20, 20, 58, 69)
    cropped_im = im.crop(crop_rectangle)
    cropped_im.show()
    #----

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue


        candidates.add(r['rect'])

    '''regions[:10]
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()'''

if __name__ == "__main__":
    main()