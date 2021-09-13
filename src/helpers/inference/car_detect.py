import re

import numpy as np
import pytesseract
import cv2


class CarDetector:

    def __init__(self, predictor0, predictor1, predictor2):
        self.predictor0 = predictor0
        self.predictor1 = predictor1
        self.predictor2 = predictor2

    @staticmethod
    def crop_img(feature_test, outputs):
        boxes = outputs["instances"].pred_boxes
        boxes = list(boxes)[0].detach().cpu().numpy()
        x, y, w, h = int(boxes[0]), int(boxes[1]), int(boxes[2] - boxes[0]), int(boxes[3] - boxes[1])
        img_ = feature_test.astype(np.uint8)
        return img_[y:y + h, x:x + w]

    def predict(self, feature_test):
        classes = {0: "Toyota", 1: "Ford", 2: "Honda", 3: "Nissan", 4: "Mercedes-Benz"}
        predictions0 = self.predictor0(feature_test)  # vehiclemodel
        predictions1 = self.predictor1(feature_test)  # PLmodel
        predictions2 = self.predictor2(feature_test)  # carMakemodel
        # print(predictions0)
        outputs = predictions0
        boxes = outputs["instances"].pred_boxes
        cropped_car = self.crop_img(feature_test, outputs)

        boxes = predictions1["instances"].pred_boxes
        boxes = list(boxes)[0].detach().cpu().numpy()
        x, y, w, h = int(boxes[0]), int(boxes[1]), int(boxes[2] - boxes[0]), int(boxes[3] - boxes[1])
        img_ = feature_test.astype(np.uint8)
        crop_img = img_[y:y + h, x:x + w]

        detect = predictions2['instances'].pred_classes.tolist()
        car_make_classes = [classes[i] for i in detect]
        try:
            car_make_classes = car_make_classes[0]
        except Exception:
            car_make_classes = "unknown"

        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # convert to grey scale

        # perfrom bitwise not to flip image to black text on white background
        crop_img = cv2.bitwise_not(crop_img)
        text = pytesseract.image_to_string(crop_img, lang='eng')
        clean_text = re.sub('[\W_]+', '', text)
        if clean_text == '':
            clean_text = 'unknown'
        else:
            clean_text = clean_text
        license_plate = clean_text
        return cropped_car, car_make_classes, license_plate
