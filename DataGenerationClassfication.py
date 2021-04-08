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
                             'sofa', 'tvmonitor']
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
        self.channels = 3
        self.indexes = np.arange(len(self.labels))
        self.label_dim = 20
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
            orig_width = int(image.shape[1])
            orig_height = int(image.shape[0])
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            target_box = random.choice(bboxes)
            bboxes.remove(target_box)

            label_vec = np.zeros(self.label_dim, dtype=np.float)
            label_vec[target_box[4]] = 1.0

            image_copy = result.copy()
            obj = image_copy[target_box[1]:target_box[3], target_box[0]:target_box[2]]
            for bbox in bboxes:
                cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), -1)
            result[target_box[1]:target_box[3], target_box[0]:target_box[2]] = obj

            if target_box[2] - target_box[0] > target_box[3] - target_box[1]:
                window_xmin = target_box[0]
                window_xmax = target_box[2]
                width = window_xmax - window_xmin
                margin_y = int((width - (target_box[3] - target_box[1])) / 2)
                window_ymin = max(target_box[1] - margin_y, 0)
                window_ymax = min(target_box[3] + margin_y, orig_height)
            else:
                window_ymin = target_box[1]
                window_ymax = target_box[3]
                height = window_ymax - window_ymin
                margin_x = int((height - (target_box[2] - target_box[0])) / 2)
                window_xmin = max(target_box[0] - margin_x, 0)
                window_xmax = min(target_box[2] + margin_x, orig_width)

            window = result[window_ymin:window_ymax, window_xmin:window_xmax]
            window = cv2.resize(window, (self.image_width, self.image_height))
            if np.random.randint(0, 100) <= 50:
                window = cv2.flip(window, 1)
            window = np.array(window, dtype=np.float)
            window /= 255.0
            #window = np.reshape(window, (window.shape[0], window.shape[1], 1))
            x[i] = window

            y[i] = label_vec

        return x, y