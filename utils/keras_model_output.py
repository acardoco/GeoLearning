from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img

import numpy as np
import cv2


def show_out():
    model = load_model('my_model_v4.h5')

    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    image_path = 'pruebas\\imag\\parking2.png'
    orig = cv2.imread(image_path)
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(40,40))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    preds = model.predict_classes(image)
    probabilities = model.predict_proba(image)
    print('Predicted:', preds)



    inID = preds[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    print('Classes: ', inv_map)
    print('prob: ', probabilities[0])

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))






