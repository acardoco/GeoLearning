'''
#***************************************
#***************************************
# Fine tuning. author: acardoco
#***************************************
#***************************************
'''

from keras import applications
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense

import numpy as np
import cv2

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 48, 48

train_data_dir = 'data\\train'
validation_data_dir = 'data\\validate'
nb_train_samples = 264
nb_validation_samples = 90
epochs = 50
batch_size = 16

num_classes = len(np.load('class_indices.npy').item())

def train_modelo():
    # build the VGG16 network
    model_vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(inputs=model_vgg.input, outputs=top_model(model_vgg.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    eval = model.evaluate_generator(validation_generator, 600)
    print(model.metrics_names)
    print(eval)

    model.save('transfer_learning.h5')

def predict_modelo():

    model = load_model('fine_tuning.h5')

    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    image_path = 'pruebas\\imag\\test_no_piscina_2.jpg'
    orig = cv2.imread(image_path)
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(48, 48))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    prob_class = preds.argmax(axis=-1)
    #probabilities = model.predict_proba(image)
    print('Predicted:', preds)

    inID = prob_class[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    print('inv_map: ', inv_map)
    print('prob: ', preds[0])
    print('inID: ', inID)
    print('prob: ', prob_class)



    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))

 #***********************************
 #***********************************
#train_modelo()
predict_modelo()