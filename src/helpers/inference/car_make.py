from src.utils.category_output import LabelCategory


class CarMake:

    def __init__(self, model):
        self.model = model

    def predict(self, feature_test):
        scores = self.model.predict(feature_test)
        output_make = LabelCategory()
        output = output_make.category_output_make(scores)
        return output
