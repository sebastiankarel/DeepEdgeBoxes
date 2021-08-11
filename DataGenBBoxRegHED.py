import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import random


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_dir, labels_dir, batch_size, image_width, image_height):
        self.class_labels = ['person', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bird',
                             'cat', 'cow', 'dog', 'horse', 'sheep', 'bottle', 'chair', 'diningtable', 'pottedplant',
                             'sofa', 'tvmonitor', 'none']
        self.batch_size = batch_size
        self.labels_dir = labels_dir
        labels = []
        for file_name in os.listdir(labels_dir):
            img_file = file_name.split(".")[0] + ".jpg"
            if os.path.isfile(os.path.join(images_dir, img_file)):
                image = cv2.imread(os.path.join(images_dir, img_file))
                if image is not None:
                    labels.append(file_name.split(".")[0])
        self.labels = labels
        self.images_dir = images_dir
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
            image = cv2.imread(os.path.join(self.images_dir, f + ".jpg"))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
            orig_width = int(image.shape[1])
            orig_height = int(image.shape[0])

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

            # Create object class window
            # Select random target object
            target_box = random.choice(bboxes)
            bboxes.remove(target_box)

            # Find image region with object (bbox + margin)
            if target_box[2] - target_box[0] > target_box[3] - target_box[1]:
                margin = int(float(target_box[2] - target_box[0]) * 0.2)
                window_xmin = max(target_box[0] - margin, 0)
                window_xmax = min(target_box[2] + margin, orig_width)
                window_ymin = max(target_box[1] - margin, 0)
                window_ymax = min(target_box[3] + margin, orig_height)
                cutout = image[window_ymin:window_ymax, window_xmin:window_xmax]
            else:
                margin = int(float(target_box[3] - target_box[1]) * 0.2)
                window_ymin = max(target_box[1] - margin, 0)
                window_ymax = min(target_box[3] + margin, orig_height)
                window_xmin = max(target_box[0] - margin, 0)
                window_xmax = min(target_box[2] + margin, orig_width)
                cutout = image[window_ymin:window_ymax, window_xmin:window_xmax]

            # Place object randomly on larger black background
            if cutout.shape[1] > cutout.shape[0]:
                window_size = int(float(cutout.shape[1]) * float(np.random.randint(1, 4)))
            else:
                window_size = int(float(cutout.shape[0]) * float(np.random.randint(1, 4)))
            x_offset = 0
            if window_size - cutout.shape[1] > 0:
                x_offset = np.random.randint(0, window_size - cutout.shape[1])
            y_offset = 0
            if window_size - cutout.shape[0] > 1:
                y_offset = np.random.randint(0, window_size - cutout.shape[0])
            window = np.zeros((window_size, window_size, self.channels))
            window[y_offset:(y_offset + cutout.shape[0]), x_offset:(x_offset + cutout.shape[1])] = cutout

            xmin = float(x_offset + margin) / float(window_size)
            ymin = float(y_offset + margin) / float(window_size)
            xmax = float(x_offset + (target_box[2] - target_box[0]) - margin) / float(window_size)
            ymax = float(y_offset + (target_box[3] - target_box[1]) - margin) / float(window_size)

            window = cv2.resize(window, (self.image_width, self.image_height))
            window = np.array(window, dtype=np.float)
            window /= 255.0
            window = np.reshape(window, (window.shape[0], window.shape[1], 1))

            x[i] = window
            y[i] = (xmin, ymin, xmax, ymax)

        return x, y