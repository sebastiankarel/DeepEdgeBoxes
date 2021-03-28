import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et


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
            scale = np.random.randint(2, self.max_scale)
            image_width = scale * self.window_width
            image_height = scale * self.window_height

            image = cv2.imread(os.path.join(self.images_dir, f + ".jpg"))
            orig_width = float(image.shape[1])
            orig_height = float(image.shape[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(cv2.resize(image, (image_width, image_height)), dtype=np.float)
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
            image = image / 255.0

            window_xmin = np.random.randint(0, image_width - self.window_width)
            window_ymin = np.random.randint(0, image_height - self.window_height)
            window_xmax = window_xmin + self.window_width
            window_ymax = window_ymin + self.window_height

            window_area = float(self.window_width * self.window_height)

            x[i] = image[window_ymin:window_ymax, window_xmin:window_xmax]

            avg_iou = 0.0
            num_boxes = 0
            tree = et.parse(os.path.join(self.labels_dir, f + ".xml"))
            root = tree.getroot()
            for bndbox in root.findall("./object/bndbox"):
                xmin = int((float(bndbox.find('xmin').text) / orig_width) * float(image_width))
                ymin = int((float(bndbox.find('ymin').text) / orig_height) * float(image_height))
                xmax = int((float(bndbox.find('xmax').text) / orig_width) * float(image_width))
                ymax = int((float(bndbox.find('ymax').text) / orig_height) * float(image_height))
                x_overlap = max(0, min(window_xmax, xmax) - max(window_xmin, xmin))
                y_overlap = max(0, min(window_ymax, ymax) - max(window_ymin, ymin))
                intersection = x_overlap * y_overlap
                if intersection > 0:
                    bbox_area = (xmax - xmin) * (ymax - ymin)
                    iou = float(intersection) / float((bbox_area + window_area - intersection))
                    avg_iou += iou
                    num_boxes += 1

            y[i] = avg_iou / float(num_boxes)

        return x, y