import os
import sys
import inspect
import traceback

import cv2
import numpy as np
from warnings import filterwarnings, simplefilter

import tensorflow as tf
from flask import Flask, request, jsonify

from helpers.inference.car_detect import CarDetector
from trained_models.detectors.loadModel import loadModel
from utils.preprocess import process_test
from responses.api import api_response
from helpers.inference.car_make import CarMake
from helpers.inference.car_type import CarType

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)
print(sub_dir)
sys.path.insert(0, sub_dir)

filterwarnings("ignore")
simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

model_make = tf.keras.models.load_model(os.path.join(sub_dir, 'trained_models/classifiers/CarMakeClassifierModel'))
model_type = tf.keras.models.load_model(os.path.join(sub_dir, 'trained_models/classifiers/CarTypeClassifierModel'))
detector0, detector1, detector2 = loadModel().load()


@app.route("/")
def home_index():
    return jsonify(success=True)


@app.route("/image_tags", methods=['GET', 'POST'])
def index():
    output = []

    if request.method == 'POST':
        row = 224
        cox = 224
        try:
            # import ipdb; ipdb.set_trace()
            file = request.files['image']
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            detected_car, car_make, license_plate = CarDetector(detector0, detector1, detector2).predict(image)
            # filename = str(uuid.uuid4().fields[-1]) + '.jpg'
            processed_image, extracted_features = process_test(detected_car, row, cox)
            out_make = CarMake(model_make).predict(extracted_features)
            if out_make == car_make:
                out_make = out_make
            else:
                out_make = car_make
            out_type = CarType(model_type).predict(extracted_features)
            output = {
                "type": out_type,
                "make": out_make,
                "licence_plate": license_plate
            }
            true_response = api_response.ApiResponse(output)
            return jsonify(true_response.true_output()), 200

        except Exception as error:
            false_response = api_response.ApiResponse(output)
            # return jsonify(false_response.error_output()), 400
            return jsonify(traceback.format_exc())
    else:
        false_response = api_response.ApiResponse(output)
        return jsonify(false_response.error_req()), 404


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
