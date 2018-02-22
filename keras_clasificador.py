#***************************************
#***************************************
# Multiclasificador para piscina, rotonda y parking
#***************************************
#***************************************

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import numpy as np 

#***************************************
# Dimensiones generales de las imagenes
#***************************************
img_width, img_height = 48, 48

#***************************************
# Parámetros generales
#***************************************
train_data_dir = 'data\\data_augmentation_output\\train'
validation_data_dir = 'data\\data_augmentation_output\\validate'
nb_train_samples = 2000
nb_validation_samples = 600
epochs = 50
batch_size = 16
num_classes = len(np.load('class_indices.npy').item())

#***************************************
# Ajusta las dimensiones en función del orden en el que aparezca el "canal"
#***************************************
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#***************************************
# Modelo creado con Keras con 4 Convs, 2 MaxPools y una FC al final
#***************************************

model = Sequential()

# 32 filtros/kernels con tamaño 3x3
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
# Esta capa ayuda a evitar overfitting
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,  activation='sigmoid'))

#***************************************
# Se indica la función de error, optimizador y métrica a mostrar
# "categorical" es para indicar que va a haber más de dos clases
#***************************************
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#***************************************
# Como es un dataset de imágenes se emplea una clase especial de Keras que permite, entre otras cosas, DataAugmentation
#***************************************

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Se reescala las imágenes para pasarlas de que valgan entre 0-255 (RGB) a que tomen valores entre 0-1
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Se "inyectan" los datasets
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
#***************************************
# Aquí se compila y entrena el modelo creado
#***************************************
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#***************************************
# Se evaluan los resultados
#***************************************
eval = model.evaluate_generator(validation_generator, 600)
print(model.metrics_names)
print(eval)


#***************************************
# Se guarda el modelo. Resultados última ejecución:
'''
40x40:
['loss', 'acc']
[0.07733457211566154, 0.97362869198312241]

48x48:
['loss', 'acc']
[0.11062831397233178, 0.97151898734177211]
'''
#***************************************
model.save('my_model_v4_48x48.h5')


