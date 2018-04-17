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
import io

import core.RoI.SelectiveSearch.selective_search_cv2 as selective_search


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    data = {"success": False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):

            image = flask.request.files["image"].read() #ERROR EN SERVIDOR: TENGO QUE GUARDARLA O SI NO VA A CASCAR !!!
            image = Image.open(io.BytesIO(image))

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
	app.run()