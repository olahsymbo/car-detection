import inspect
import os
import sys

import numpy as np

app_path = inspect.getfile(inspect.currentframe())
sub_dir = os.path.realpath(os.path.dirname(app_path))
main_dir = os.path.dirname(sub_dir)
sys.path.insert(0, str(main_dir))

from src.utils.preprocess import process


class CreateTrainingDataset:
    def __init__(self, data_type, all_folders, all_labels, row, cox):
        self.data_type = data_type
        self.all_folders = all_folders
        self.all_labels = all_labels
        self.row = row
        self.cox = cox

    def make_dataset(self):
        dataset = []
        label = []
        for i in range(len(self.all_folders)):
            for imgName in os.listdir(os.path.join(main_dir, self.data_type + "/" + self.all_folders[i])):
                path = os.path.join(main_dir, self.data_type + "/" + self.all_folders[i] + "/" + imgName)
                if path.endswith('jpg') or path.endswith('png') or path.endswith('jpeg'):
                    print("Processing.... ", path)
                    img = process(path, self.row, self.cox)
                    dataset.append(np.array(img))
                    label.append(np.array(self.all_labels[i]))
        return dataset, label
