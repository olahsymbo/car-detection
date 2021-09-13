import os
import sys
from pathlib import Path
import inspect
app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)

ROOT_DIR = Path(__file__).parent.parent.parent
print(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import classification_report
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D

from src.utils.image_loader import CreateTrainingDataset
from src.utils.conf_matr import plot_confusion_matrix

DATA_FOLDER = 'images/types'
saloon_folder = 'saloon_photo'
suv_folder = 'suv_photo'
truck_folder = 'truck_photo'
van_folder = 'van_photo'

rows, cols, band = 224, 224, 3

all_folders = [saloon_folder, suv_folder, truck_folder, van_folder]

# labels for images
label_saloon = 'saloon'
label_suv = 'suv'
label_truck = 'truck'
label_van = 'van'

all_labels = [label_saloon, label_suv, label_truck, label_van]

# create training dataset with label
create_data = CreateTrainingDataset(DATA_FOLDER, all_folders, all_labels, rows, cols)
dateset, data_label = create_data.make_dataset()

# Splitting the feature and label for train dataset
matrix_data = [i for i in dateset]
print(len(matrix_data))
matrix_data = np.array(matrix_data).reshape(-1, rows, cols, band)
data_label = [i for i in data_label]
data_label = np.array(data_label).astype(str)

feature_train, feature_test, y_train, y_test = train_test_split(matrix_data, data_label,
                                                                test_size=0.30, random_state=1000)

print("Shape of training set", y_train.shape)
print("Shape of testing set", y_test.shape)

number = preprocessing.LabelEncoder()
obj_fit = number.fit(y_train)
labels_train = obj_fit.transform(y_train)
labels_test = obj_fit.transform(y_test)

y_train = to_categorical(labels_train)
y_test = to_categorical(labels_test)
print(y_test.shape)

# #### Using transfer learning with Inception V3 pretrained model
n_classes = y_train.shape[1]


base_model = VGG16(weights='imagenet', include_top=False)
# base_model = ResNet50(weights='imagenet', include_top=False)
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))
model.layers[0].trainable = False
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(feature_train, y_train, validation_split=0.2, epochs=30, batch_size=60)

scores = model.predict(feature_test)

model.save('trained_models/classifiers/CarTypeClassifierModel')


print('classification report\n',classification_report(np.argmax(y_test, axis=1), np.argmax(scores, axis=1)))
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(scores, axis=1))
labels = ["saloon",  "suv",  "truck",  "van"]


plot_confusion_matrix(cm,labels,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True)
