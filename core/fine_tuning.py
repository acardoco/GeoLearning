'''
#***************************************
#***************************************
# Fine tuning. author: acardoco
#***************************************
#***************************************
'''

from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense

import matplotlib.pyplot as plt
import numpy as np
import cv2

#***************************************
# Parámetros generales
#***************************************
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width, img_height = 48, 48

train_data_dir = 'data\\data_augmentation_output\\train'
validation_data_dir = 'data\\data_augmentation_output\\validate'
modelo_save_dir = 'fine_tuning_18_48x48_parking.h5'
nb_train_samples = 2000
nb_validation_samples = 600
epochs = 50
batch_size = 16
num_classes = len(np.load('class_indices.npy').item())

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
# Función que entrena y usa Transfer Learning - Fine Tuning para hacer una arquitectura bottleneck feature
'''
 modelo fine tuning = model_vgg(-top_layer) + ( top_layer creada manualmente + pesos entrenados con Transfer-Learning )
 
 En pocas palabras: a diferencia de T-F que parte desde 0, con F-T se parte de los pesos previamente entrenados con T-F: 
    por eso "refinamos" el modelo
'''
#***************************************
def train_modelo():

    # se carga el modelo VGG16
    model_vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # FC que se colocará en el top del modelo
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='sigmoid'))

    # Para hacer Fine-Tuning hay que partir de un clasificador ya entrenado: en este caso se emplea el entrenado con Trasnfer-Learning y VGG16
    top_model.load_weights(top_model_weights_path)

    # Se instancia una clase tipo "Model", añadiendo top_model encima de model_vgg
    model = Model(inputs=model_vgg.input, outputs=top_model(model_vgg.output))

    # Se "congelan" las 15 primeras capas para que no sean entrenadas:
    # 18 capas tiene vgg16 sin las FC del tope, por lo que habrá que quitar las 3 ultimas que son conv para hacer fine-tuning (18 - 3 = 15)
    for layer in model.layers[:18]:
        layer.trainable = False

    # Se compila
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
                  metrics=['accuracy'])

    # Se "inyectan" los datasets
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


    # Se entrena el modelo con Fine-Tuning
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    eval = model.evaluate_generator(validation_generator, 600)

    print('Resultado', eval)

    model.save(modelo_save_dir)

    print_train_sumary(history)
    '''
    Resultado sin DataAug previo [0.12555930473659541, 0.94444444368945224]
    Resultado con DataAug previo[0.11911813156126431, 0.95822784810126582]
    '''

    '''18 capas congelas (Transfer Learning):
    Resultado [0.0055666465254670203, 1.0]
    
    15 capas congelas con rango de aprendizaje muy bajo y SGD e-4 (Fine Tunning):
    Resultado [0.11849022399826352, 0.97531645569620251]
    
    15 capas congelas con rango de aprendizaje muy bajo y SGD e-5 (Fine Tunning):
    Resultado [0.030713025634991935, 0.99198312236286923]
    
    15 capas congelas con rango de aprendizaje muy bajo y SGD e-6 (Fine Tunning):
    Resultado [0.090149949951365618, 0.98829113924050638]
    
    **DATASET V2:
    
    15 capas congelas con rango de aprendizaje muy bajo y SGD e-6 (Fine Tunning) Y dataset_v2 con 68x68:
    Resultado [0.077296504126468796, 0.99103375527426163]
    
    15 capas congelas con rango de aprendizaje muy bajo y SGD e-5 (Fine Tunning) Y dataset_v2 con 48x48:
    Resultado [0.052187373268115178, 0.99029535864978901]
    
    18 CAPAS congelas con rango de aprendizaje muy bajo y SGD e-4 (Fine Tunning) Y dataset_v2 con 48x48:
    Resultado [0.036398688509460361, 0.99008438818565403]
    
    FIN DATASET V2
    
    18 CAPAS congelas con rango de aprendizaje muy bajo y SGD e-5 (Fine Tunning) y dataset original con parking modificado (fine_tuning_18_48x48_parking.h5):
    Resultado [0.1500440054625479, 0.97341772151898731]
    '''
#***************************************
# Función-prueba para predecir imagenes
#***************************************
def predict_modelo():

    model = load_model(modelo_save_dir)

    # Se cargan las clases
    class_dictionary = np.load('class_indices.npy').item()

    image_path = 'pruebas\\imag\\piscina.jpg'
    orig = cv2.imread(image_path)
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(img_width, img_height))
    image = img_to_array(image)

    image = image / 255

    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    #necesario hacer esto para obtener las prob. ya que es un modelo de la clase "Model" y no tiene otros métodos de la clase "Sequential"
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

    '''for i,layer in enumerate(model.layers):
        print(i, layer.name, layer.output_shape,layer.trainable)'''

 #***********************************
 #***********************************

#***************************************
# Funciones a ejecutar
#***************************************
train_modelo()
predict_modelo()