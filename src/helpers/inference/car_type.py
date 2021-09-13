from src.utils.category_output import LabelCategory


class CarType:

    def __init__(self, model):
        self.model = model

    def predict(self, feature_test):
        scores = self.model.predict(feature_test)
        output_type = LabelCategory()
        output = output_type.category_output_type(scores)
        return output
