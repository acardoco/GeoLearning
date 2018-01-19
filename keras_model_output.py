from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('my_model_v2.h5')

img_path = 'pruebas\\mar.jpg'
img = image.load_img(img_path, target_size=(40, 40))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

preds = model.predict_classes(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', preds)




