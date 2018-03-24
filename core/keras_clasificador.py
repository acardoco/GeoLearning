#***************************************
#***************************************
# Multiclasificador para piscina, rotonda y parking
#***************************************
#***************************************
import matplotlib.pyplot as plt
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
train_data_dir = 'C:\\Users\Andrés\Documents\\UC3M\TFM\GeoLearning\datos\data_v3\set_80x80_2\\train'
validation_data_dir = 'C:\\Users\Andrés\Documents\\UC3M\TFM\GeoLearning\datos\data_v3\set_80x80_2\\validate'
nb_train_samples = 1604 #991
nb_validation_samples = 455 #348
epochs = 80 #50
batch_size = 32 #32
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

#hasta aquí capas iniciales
'''
model.add(Conv2D(128, kernel_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pooling_size))

model.add(Conv2D(256, (1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(256, (1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))

# hasta aquí "algunas capas"

model.add(Conv2D(512, (1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(512, (1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,1)))'''

model.add(Flatten())
model.add(Dense(256, activation='relu')) #256
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

'''train_datagen = ImageDataGenerator(
    rescale=1. / 255)'''
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.4,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

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


DATA_v3 
['loss', 'acc']
[0.11944854707336736, 0.9626527613996706] con 80x80 sin Aug y categorical_cross_entropy
[0.3636461944851454, 0.90222222222222226] con 80x80 y Aug(3000_450) y categorical_cross_entropy
[0.34997867314012615, 0.89837673324942524] con 80x80, 80 epochs y Aug(2300_325 compensado) y categorical_cross_entropy
    DATA_v3 set_80x80_2
    (+)[0.14274147466839379, 0.95478723508246399] con 80x80, 80 epochs sin Aug y categorical_cross_entropy
    (*) [0.10390177214508282, 0.96542553315771384] igual pero con + PARAMETROS en ImageDataAugmentation
    [0.19170924897444375, 0.93085106863620437] igual pero con más capas, 1024-FC 64 batch y 100 epochs
    [0.3818404035682374, 0.86170212897848575] 80 epoch, 32 batch y + más capas y 1024-FC
    [0.14657957505949951, 0.96010638437372575] (*) en "my_model_dv3_80x80_2"
    [0.31570550615800186, 0.91755319280827297] (*) con más capas
    [0.63741721928119655, 0.81648936281812956] (*) con más capas pero 128-256-512-1024FC 
    [0.24499686194639037, 0.9196662303664922] (*) con más capas pero 128-256-512-256FC y fuera datos de dataset_v2
    [0.17947033210129301, 0.94049956369982546] (*) y fuera datos de dataset_v2
    [0.16494690282550209, 0.94039048865619546] (+) y fuera datos de dataset_v2
    (-)[0.098912827930487224, 0.95397489781035538] con datos de DataAug-rotonda de datasetv2 y 80 epochs ... -> en selecSearch no pilla las rotondas
    [0.98016709552524484, 0.8158995858171495] (-) con más capas
    (.)[0.15047867573473439, 0.93776824051602203] con data Aug a rotonda "my_model_dv3_80x80_2_prueba" -> ya pilla rotondas
    [0.19156197065431543, 0.94420600889706308] (.) con algunas capas  "my_model_dv3_80x80_2-2"
    [0.48811058192513884, 0.87124463605650504] (.) con algunas capas   y 150 epochs
    [0.21628767510860072, 0.91630901336797832] (.) con algunas capas y 100 epochs
    (^)[0.1870488715519269, 0.94505494541191792] con algunas capas y algunos ejemplos en rotonda eliminados (71: cosas borrosas o solos partes de rotondas)
    [0.10224502108150973, 0.95384615411470219] (^) 48x48 "my_model_dv3_80x80_2_prueba_3"
'''
#***************************************
model.save('C:\\Users\Andrés\Documents\\UC3M\TFM\GeoLearning\modelos\my_model_dv3_80x80_2_prueba_3.h5')


