import cv2
import numpy as np


def process(imageName, row, cox):
    output = None
    # check the format of image
    if imageName.endswith('jpg') or imageName.endswith('png') or imageName.endswith('jpeg'):
        # read the image in grayscale
        grayscaleImage = cv2.imread(imageName) / 255
        # grayscaleImage = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE) / 255

        output = cv2.resize(grayscaleImage, (row, cox))

    return output


def process_test(image, row, cox):
    # img_str = imageName.stream.read()
    # print(len(np.fromstring(imageName.read(), np.uint8)))
    img_resize = cv2.resize(image, (row, cox))
    feature_test = np.array(img_resize).reshape(-1, row, cox, 3)
    return img_resize, feature_test
