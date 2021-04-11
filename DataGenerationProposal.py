import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import random


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, edge_images_dir, labels_dir, batch_size, window_width, window_height):
        self.batch_size = batch_size
        self.labels_dir = labels_dir
        labels = []
        for file_name in os.listdir(labels_dir):
            file_name = file_name.split(".")[0]
            if os.path.isfile(os.path.join(edge_images_dir, file_name + ".jpg")):
                image = cv2.imread(os.path.join(edge_images_dir, file_name + ".jpg"))
                if image is not None:
                    labels.append(file_name.split(".")[0])
        self.labels = labels
        self.images_dir = edge_images_dir
        self.indexes = np.arange(len(self.labels))
        self.window_height = window_height
        self.window_width = window_width
        self.max_scale = 6
        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        labels_temp = [self.labels[k] for k in indexes]
        x, y = self.__data_generation(labels_temp)
        return x, y

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.labels))
        np.random.shuffle(self.labels)

    def __data_generation(self, labels_temp):
        x = np.empty((self.batch_size, self.window_height, self.window_width, 1))
        y = np.empty((self.batch_size, 1), dtype=np.float)
        for i, f in enumerate(labels_temp):
            image = cv2.imread(os.path.join(self.images_dir, f + ".jpg"))
            orig_width = int(image.shape[1])
            orig_height = int(image.shape[0])

            # Generate edge image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            no_sigma = image.copy()
            low_sigma = cv2.GaussianBlur(image, (3, 3), 0)
            high_sigma = cv2.GaussianBlur(image, (5, 5), 0)
            sigma = 0.33
            v = np.median(image)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            no_sigma = cv2.Canny(no_sigma, lower, upper)
            low_sigma = cv2.Canny(low_sigma, lower, upper)
            high_sigma = cv2.Canny(high_sigma, lower, upper)
            result = np.dstack((no_sigma, low_sigma, high_sigma))

            # Read ground truth
            bboxes = []
            tree = et.parse(os.path.join(self.labels_dir, f + ".xml"))
            root = tree.getroot()
            for obj in root.findall("./object"):
                label_name = obj.find('name').text
                label = self.class_labels.index(label_name)
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bboxes.append((xmin, ymin, xmax, ymax, label))

            # Cut out random window
            window = result

            window = cv2.resize(window, (self.image_width, self.image_height))
            window = np.array(window, dtype=np.float)
            window /= 255.0

            x[i] = window
            y[i] = 0.0

        return x, y