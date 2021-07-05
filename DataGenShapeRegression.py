import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import random
import EdgeDetection as ed


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_dir, labels_dir, batch_size, image_width, image_height, use_augmentation, multi_channel):
        self.batch_size = batch_size
        self.labels_dir = labels_dir
        self.multi_channel = multi_channel
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
        if multi_channel:
            self.channels = 3
        else:
            self.channels = 1
        self.indexes = np.arange(len(self.labels))
        self.label_dim = 2
        self.use_augmentation = use_augmentation
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
            result = ed.auto_canny(image, self.multi_channel)
            if not self.multi_channel:
                result = np.reshape(result, (result.shape[0], result.shape[1], 1))

            # Read ground truth
            bboxes = []
            tree = et.parse(os.path.join(self.labels_dir, f + ".xml"))
            root = tree.getroot()
            for obj in root.findall("./object"):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bboxes.append((xmin, ymin, xmax, ymax))

            # Select random target object
            target_box = random.choice(bboxes)
            bboxes.remove(target_box)

            # Compute box dimensions
            box_width = target_box[2] - target_box[0]
            box_height = target_box[3] - target_box[1]

            # Cut out image region with random margin
            if box_width > box_height:
                x_offset = float(box_width) * 0.2
                x_min_margin = int(x_offset * np.random.random())
                x_max_margin = int(x_offset) - x_min_margin
                window_xmin = max(target_box[0] - x_min_margin, 0)
                window_xmax = min(target_box[2] + x_max_margin, orig_width)
                window_height = window_xmax - window_xmin  # window height = window width
                y_margin = window_height - box_height
                y_min_margin = int(float(y_margin) * np.random.random())
                y_max_margin = y_margin - y_min_margin
                window_ymin = max(target_box[1] - y_min_margin, 0)
                window_ymax = min(target_box[3] + y_max_margin, orig_height)
            else:
                y_offset = float(box_height) * 0.2
                y_min_margin = int(y_offset * np.random.random())
                y_max_margin = int(y_offset) - y_min_margin
                window_ymin = max(target_box[1] - y_min_margin, 0)
                window_ymax = min(target_box[3] + y_max_margin, orig_height)
                window_width = window_ymax - window_ymin  # window width = window height
                x_margin = window_width - box_width
                x_min_margin = int(float(x_margin) * np.random.random())
                x_max_margin = x_margin - x_min_margin
                window_xmin = max(target_box[0] - x_min_margin, 0)
                window_xmax = min(target_box[2] + x_max_margin, orig_width)
            cutout = result[window_ymin:window_ymax, window_xmin:window_xmax]

            # Reshape single channel image
            if not self.multi_channel:
                cutout = np.reshape(cutout, (cutout.shape[0], cutout.shape[1], 1))

            # Pad image to make square
            if cutout.shape[1] > cutout.shape[0]:
                window = np.zeros((cutout.shape[1], cutout.shape[1], self.channels))
                margin = int(float(cutout.shape[1] - cutout.shape[0]) / 2.0)
                window[margin:(margin + cutout.shape[0]), :, :] = cutout
            elif cutout.shape[1] < cutout.shape[0]:
                window = np.zeros((cutout.shape[0], cutout.shape[0], self.channels))
                margin = int(float(cutout.shape[0] - cutout.shape[1]) / 2.0)
                window[:, margin:(margin + cutout.shape[1]), :] = cutout
            else:
                window = cutout

            # Compute relative width and height
            label_x = float(box_width) / float(window.shape[1])
            label_y = float(box_height) / float(window.shape[0])

            window = cv2.resize(window, (self.image_width, self.image_height))
            window = np.array(window, dtype=np.float)
            if not self.multi_channel:
                window = np.reshape(window, (window.shape[0], window.shape[1], 1))
            window /= 255.0

            label_vec = np.zeros(self.label_dim, dtype=np.float)
            label_vec[0] = label_x
            label_vec[1] = label_y

            x[i] = window
            y[i] = label_vec

        return x, y