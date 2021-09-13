import numpy as np


class LabelCategory:

    def __init__(self):
        self.car_make = ["Benz", "Honda", "Hyundai", "Nissan", "Toyota"]
        self.car_type = ["saloon", "suv", "truck", "van"]

    def category_output_make(self, prediction):
        result = self.car_make[np.argmax(prediction)]
        return result

    def category_output_type(self, prediction):
        result = self.car_type[np.argmax(prediction)]
        return result
