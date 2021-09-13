import os
import sys
import inspect
from pathlib import Path

import numpy as np
app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import classification_report

from src.utils.image_loader import CreateTrainingDataset

DATA_FOLDER = 'images'
BENZ_FOLDER = 'Benz_photo'
HONDA_FOLDER = 'Honda_photo'
HYUNDAI_FOLDER = 'Hyundia_photo'
NISSAN_FOLDER = 'Nissan_photo'
TOYOTA_FOLDER = 'Toyota_photo'

rows, cols, band = 224, 224, 3

all_folders = [BENZ_FOLDER, HONDA_FOLDER, HYUNDAI_FOLDER, NISSAN_FOLDER,
               TOYOTA_FOLDER]

# labels for images
label_Benz = 'Benz'
label_Honda = 'Honda'
label_Hyundai = 'Hyundai'
label_Nissan = 'Nissan'
label_Toyota = 'Toyota'

all_labels = [label_Benz, label_Honda, label_Hyundai, label_Nissan,
              label_Toyota]

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

# Using transfer learning with Inception V3 pretrained model
n_classes = y_train.shape[1]
# base_model = VGG16(weights='imagenet', include_top=False)
base_model = VGG16(weights='imagenet', include_top=False)
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

# Final evaluation of the model
scores = model.predict(feature_test, verbose=0)
# print("Large CNN Accuracy: %.2f%%" % (100-scores[1]*100))

model.save('CarMakeClassifierModel')

print('classification report\n',classification_report(np.argmax(y_test, axis=1), np.argmax(scores, axis=1)))
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(scores, axis=1))
labels = ["Benz",  "Honda",  "Hyundai",  "Nissan",  "Toyota"]

from src.utils.conf_matr import plot_confusion_matrix

plot_confusion_matrix(cm,labels,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True)
