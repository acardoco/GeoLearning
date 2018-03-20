#***************************************
#***************************************
# Multiclasificador para piscina, rotonda y parking
#***************************************
#***************************************
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import numpy as np

#***************************************
# Dimensiones generales de las imagenes
#***************************************
img_width, img_height = 80, 80

#***************************************
# Parámetros generales
#***************************************
train_data_dir = 'C:\\Users\Andrés\Documents\\UC3M\TFM\GeoLearning\datos\data_v3\set_80x80\\train'
validation_data_dir = 'C:\\Users\Andrés\Documents\\UC3M\TFM\GeoLearning\datos\data_v3\set_80x80\\validate'
nb_train_samples = 991
nb_validation_samples = 348
epochs = 50
batch_size = 32
num_classes = len(np.load('outputs_de_modelos/class_indices.npy').item())
kernel_size = (3,3)
pooling_size =(3,3)

#***************************************
# Ajusta las dimensiones en función del orden en el que aparezca el "canal"
#***************************************
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#***************************************
# Función para mostrar resultados de entrenamiento
#***************************************
def print_train_sumary(history):

    plt.figure(1)

    # Acierto

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Precisión del modelo')
    plt.ylabel('precisión')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Error

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Error en modelo')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#***************************************
# Modelo creado con Keras con 4 Convs, 2 MaxPools y una FC al final
#***************************************

model = Sequential()

# 32 filtros/kernels con tamaño 3x3
model.add(Conv2D(32, kernel_size, input_shape=input_shape))
model.add(Activation('relu'))
# Esta capa ayuda a evitar overfitting
model.add(Dropout(0.5))

model.add(Conv2D(32, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pooling_size))

model.add(Conv2D(64, kernel_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pooling_size))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
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
    rescale=1. / 255)
'''train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.4,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')'''

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
history = model.fit_generator(
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

print_train_sumary(history)


#***************************************
# Se guarda el modelo. Resultados última ejecución:
'''
40x40:
['loss', 'acc']
[0.07733457211566154, 0.97362869198312241]

48x48:
['loss', 'acc']
[0.11062831397233178, 0.97151898734177211]
[0.087295719137207017, 0.97035864978902953] -- nuevas cosas añadidas al ImageDataGenerator
[0.32504198165131015, 0.87679324894514765] -- nuevas cosas añadidas al ImageDataGenerator && [kernel = (2,2) || nuevas capas con 128]
[0.19394528099850511, 0.93164556962025313] -- nuevas cosas añadidas al ImageDataGenerator_v2
[0.062006123949340776, 0.97509497733697859] -- nuevas cosas añadidas al ImageDataGenerator_v3
[0.069960000949393461, 0.97231759656652361] -- Dataset_Augmented 4000-900
[0.20118663615366064, 0.95112660944206007] -- dataset_aug_v2
[0.21565412196183703, 0.97006437768240339] -- dataset_aug_v2 56x56 90epochs
[0.1882300089876697, 0.96770386266094421] -- dataset_aug_v2 68x68
[0.16709360712583873, 0.96111583460941019] --dataset_aug_v2 56x56 40batch
[0.18661447051514896, 0.95330308237474348] --dataset_aug_v2_2000_600 30x30 32batch
[0.049814810118578613, 0.98675601552336867] --dataset original con algunas modificaciones en las imagenes del parking (48x48)

ERROR CAMBIADO A mean_squared_logarithmic_error
[0.0098129704478977457, 0.96625000000000005] -- dataset_aug_v2_3000_800 con distorsiones en imagenes


DATA_v3 con 80x80 sin Aug y categorical_cross_entropy
['loss', 'acc']
[0.11944854707336736, 0.9626527613996706]

'''
#***************************************
model.save('C:\\Users\Andrés\Documents\\UC3M\TFM\GeoLearning\modelos\my_model_dv3_80x80.h5')


