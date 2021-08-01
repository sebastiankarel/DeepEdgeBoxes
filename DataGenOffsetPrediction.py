import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import random
import EdgeDetection as ed


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_dir, labels_dir, batch_size, image_width, image_height, multi_channel, rgb):
        self.class_labels = ['person', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bird',
                             'cat', 'cow', 'dog', 'horse', 'sheep', 'bottle', 'chair', 'diningtable', 'pottedplant',
                             'sofa', 'tvmonitor', 'none']
        self.batch_size = batch_size
        self.labels_dir = labels_dir
        self.multi_channel = multi_channel
        self.rgb = rgb
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
        if multi_channel or rgb:
            self.channels = 3
        else:
            self.channels = 1
        self.indexes = np.arange(len(self.labels))
        self.label_dim = 2
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

            # Generate edge image
            result = ed.auto_canny(image, self.multi_channel, self.rgb)
            if not self.multi_channel and not self.rgb:
                result = np.reshape(result, (result.shape[0], result.shape[1], 1))

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
                cutout = result[window_ymin:window_ymax, window_xmin:window_xmax]
            else:
                margin = int(float(target_box[3] - target_box[1]) * 0.2)
                window_ymin = max(target_box[1] - margin, 0)
                window_ymax = min(target_box[3] + margin, orig_height)
                window_xmin = max(target_box[0] - margin, 0)
                window_xmax = min(target_box[2] + margin, orig_width)
                cutout = result[window_ymin:window_ymax, window_xmin:window_xmax]

            if not self.multi_channel and not self.rgb:
                cutout = np.reshape(cutout, (cutout.shape[0], cutout.shape[1], 1))

            # Place image region centred on square black background
            if cutout.shape[1] > cutout.shape[0]:
                window = np.zeros((cutout.shape[1], cutout.shape[1], self.channels))
                margin = int((window.shape[0] - cutout.shape[0]) / 2)
                window[margin:(margin + cutout.shape[0]), :, :] = cutout
            else:
                window = np.zeros((cutout.shape[0], cutout.shape[0], self.channels))
                margin = int((window.shape[1] - cutout.shape[1]) / 2)
                window[:, margin:(margin + cutout.shape[1]), :] = cutout

            # Place object randomly on larger black background
            x_offset = np.random.random()
            y_offset = np.random.random()
            new_window = np.zeros((int(window.shape[0] * 1.5), int(window.shape[1] * 1.5), self.channels))
            x_margin = int(float(window.shape[1]) * x_offset * 0.5)
            y_margin = int(float(window.shape[0]) * y_offset * 0.5)
            new_window[y_margin:(y_margin + window.shape[0]), x_margin:(x_margin + window.shape[1]), :] = window

            new_window = cv2.resize(new_window, (self.image_width, self.image_height))
            new_window = np.array(new_window, dtype=np.float)
            if not self.multi_channel and not self.rgb:
                new_window = np.reshape(new_window, (new_window.shape[0], new_window.shape[1], 1))
            new_window /= 255.0

            x[i] = new_window
            y[i] = (x_offset, y_offset)

        return x, y