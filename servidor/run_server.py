# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from PIL import Image

import json
import flask
import urllib.request as urlr

import core.RoI.SelectiveSearch.selective_search_cv2 as selective_search

# params
base = 'https://maps.googleapis.com/maps/api/staticmap?center='
parametros_comunes = '&zoom=18&format=jpg&size=400x400&maptype=satellite&key='
key = 'AIzaSyB9qW-QzzGtT2xEsJlsuLgA5TOYNJS8ogo'

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"
@app.route("/prueba", methods = ["POST"])
def prueba():

    data = {"success": False}

    content = flask.request.json
    data["candidatos"] = {"lat":"pepe", "lon":"paco","prob":"pipo", "class":"keko"}
    print(content)

    data["success"] = True

    return flask.jsonify(data)



@app.route("/predict", methods=["POST"])
def predict():

    data = {"success": False}

    # TODO recoger lat y lon en json y hacer peticion a GoogleMapsApi
    content = flask.request.json
    lat = content['lat']
    lon = content['lon']
    url_rquest = base + lat + ',' + lon + parametros_comunes + key
    IMAGE_PATH = "tmp/" + lat + '_' + lon +'.jpg'
    urlr.urlretrieve(url_rquest,IMAGE_PATH)
    print('Url to: ', url_rquest)

    # QUITAR estas 2 lineas EN LA VERSION FINAL
    # IMAGE_PATH = "ciudades\ciudad6.jpg"
    image = Image.open(IMAGE_PATH)

    #selective search
    candidates = selective_search.selec_cv2(image)

    data["candidatos"] = []

    for item in candidates:
        label = item[0]
        x, y, w, h = item[1]
        prob = item[2]
        r = {"label": label, "x": int(x), "y": int(y), "w": int(w), "h": int(h), "prob": float(prob)}
        data["candidatos"].append(r)

    data["success"] = True

    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run(host='0.0.0.0')