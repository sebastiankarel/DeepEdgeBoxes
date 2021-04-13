import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import math


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
            # Read and enlarge image to scan at different scales
            scale = np.random.randint(2, 6)
            image = cv2.imread(os.path.join(self.images_dir, f + ".jpg"))
            orig_width = image.shape[1]
            orig_height = image.shape[0]
            image = cv2.resize(image, (scale * self.window_width, scale * self.window_height))

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

            # Read ground truth and convert to scale
            bboxes = []
            tree = et.parse(os.path.join(self.labels_dir, f + ".xml"))
            root = tree.getroot()
            for obj in root.findall("./object"):
                bndbox = obj.find('bndbox')
                xmin = round((float(bndbox.find('xmin').text) / float(orig_width)) * float(result.shape[1]))
                ymin = round((float(bndbox.find('ymin').text) / float(orig_height)) * float(result.shape[0]))
                xmax = round((float(bndbox.find('xmax').text) / float(orig_width)) * float(result.shape[1]))
                ymax = round((float(bndbox.find('ymax').text) / float(orig_height)) * float(result.shape[0]))
                bboxes.append((xmin, ymin, xmax, ymax))

            # Cut out random window
            window_xmin = np.random.randint(0, (result.shape[1] - self.window_width) + 1)
            window_ymin = np.random.randint(0, (result.shape[0] - self.window_height) + 1)
            window_xmax = window_xmin + self.window_width
            window_ymax = window_ymin + self.window_height
            window = result[window_ymin:window_ymax, window_xmin:window_xmax]
            window = np.array(window, dtype=np.float)
            window /= 255.0

            # Create label as iou of most central object
            box_results = []
            for box in bboxes:
                x_overlap = max(0, min(box[2], window_xmax) - max(box[0], window_xmin))
                y_overlap = max(0, min(box[3], window_ymax) - max(box[1], window_ymin))
                intersection = x_overlap * y_overlap
                if intersection > 0:
                    box_x_center = round(box[0] + ((box[2] - box[0]) / 2.0))
                    box_y_center = round(box[1] + ((box[3] - box[1]) / 2.0))
                    window_x_center = round(window_xmin + (float(self.window_width) / 2.0))
                    window_y_center = round(window_ymin + (float(self.window_height) / 2.0))
                    center_dist = math.sqrt(math.pow(box_x_center - window_x_center, 2) + math.pow(box_y_center - window_y_center, 2))
                    window_area = float(self.window_width * self.window_height)
                    box_area = float((box[2] - box[0])) * float((box[3] - box[1]))
                    union = window_area + box_area - float(intersection)
                    iou = intersection / union
                    box_results.append((center_dist, iou))

            # Select center most bounding box and use iou as label
            if len(box_results) > 0:
                box_results.sort(key=lambda box_res: box_res[0])
            else:
                box_results.append((0.0, 0.0))

            x[i] = window
            y[i] = box_results[0][1]

        return x, y