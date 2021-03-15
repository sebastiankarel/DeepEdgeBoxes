import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_dir, label_dir, image_width, image_height, batch_size):
        self.batch_size = batch_size
        self.label_dir = label_dir
        labels = []
        for file_name in os.listdir(label_dir):
            img_file = file_name.split(".")[0] + ".jpg"
            if os.path.isfile(os.path.join(image_dir, img_file)):
                image = cv2.imread(os.path.join(image_dir, img_file))
                if image is not None:
                    labels.append(file_name.split(".")[0])
        self.labels = labels
        self.image_dir = image_dir
        self.image_width = image_width
        self.image_height = image_height
        self.channels = 1
        self.indexes = np.arange(len(self.labels))
        self.label_dim = 4
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
        x = np.empty((self.batch_size, self.image_height, self.image_width, self.channels))
        y = np.empty((self.batch_size, self.label_dim))
        for i, f in enumerate(labels_temp):
            image = cv2.imread(os.path.join(self.image_dir, f + ".jpg"))
            orig_width = float(image.shape[1])
            orig_height = float(image.shape[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(cv2.resize(image, (self.image_width, self.image_height)), dtype=np.float)
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
            x[i] = image

            tree = et.parse(os.path.join(self.label_dir, f + ".xml"))
            root = tree.getroot()
            for bndbox in root.findall("./object/bndbox"):
                xmin = float(bndbox.find('xmin').text) / orig_width
                ymin = float(bndbox.find('ymin').text) / orig_height
                xmax = float(bndbox.find('xmax').text) / orig_width
                ymax = float(bndbox.find('ymax').text) / orig_height
                y[i] = [xmin, ymin, xmax, ymax]

        return x, y