# USAGE
# python simple_request.py

# import the necessary packages
import requests

def prueba():

    JSON_URL = "http://localhost:5000/prueba"

    json = {'lat': 4.324234, 'lon': 3.42342}

    r = requests.post(JSON_URL,json=json).json()

    if r["success"]:
        print(r)
    else:
        print("Request failed")

def predict():
    # initialize the Keras REST API endpoint URL along with the input
	# image path
	KERAS_REST_API_URL = "http://localhost:5000/predict"
	IMAGE_PATH = "ciudades\ciudad6.jpg"

	# load the input image and construct the payload for the request
	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": image}


	# submit the request
	r = requests.post(KERAS_REST_API_URL, files=payload).json()


	# ensure the request was sucessful
	if r["success"]:
		# loop over the predictions and display them
		for (i, result) in enumerate(r["candidatos"]):
			print("{}. {}: {:.4f}".format(i + 1, result["label"],
				result["prob"]))

	# otherwise, the request failed
	else:
		print("Request failed")

if __name__ == "__main__":

	prueba()