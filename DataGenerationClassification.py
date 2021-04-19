import tensorflow as tf
import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import random


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, images_dir, labels_dir, batch_size, image_width, image_height, use_augmentation):
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
        self.channels = 3
        self.indexes = np.arange(len(self.labels))
        self.label_dim = 21
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

            if np.random.randint(0, 21) <= 20:
                # Create none class window
                # Create label vector
                label_vec = np.zeros(self.label_dim, dtype=np.float)
                label_vec[20] = 1.0

                # Mask out any objects with white noise
                for box in bboxes:
                    result[box[1]:box[3], box[0]:box[2]] = np.zeros((box[3] - box[1], box[2] - box[0], 3))

                # Cut out random window
                scale = np.random.randint(1, 6)
                rescaled_width = scale * orig_width
                rescaled_height = scale * orig_height
                while self.image_height >= rescaled_height or self.image_width >= rescaled_width:
                    scale += 1
                    rescaled_width = scale * orig_width
                    rescaled_height = scale * orig_height
                result = cv2.resize(result, (rescaled_width, rescaled_height))
                xmin = np.random.randint(0, rescaled_width - self.image_width)
                ymin = np.random.randint(0, rescaled_height - self.image_height)
                xmax = xmin + self.image_width
                ymax = ymin + self.image_height
                window = result[ymin:ymax, xmin:xmax]
            else:
                # Create object class window
                # Select random target object
                target_box = random.choice(bboxes)
                bboxes.remove(target_box)
                # Create label vector
                label_vec = np.zeros(self.label_dim, dtype=np.float)
                label_vec[target_box[4]] = 1.0

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

                if self.use_augmentation:
                    # Augment 50% of data points
                    if np.random.randint(0, 100) <= 50:
                        rand_val = np.random.randint(0, 3)
                        # Resize to min 50% of size
                        if rand_val == 0:
                            aspect_ration = float(cutout.shape[1]) / float(cutout.shape[0])
                            new_height = float(cutout.shape[0]) * np.random.uniform(0.5, 1.0)
                            new_width = new_height * aspect_ration
                            cutout = cv2.resize(cutout, (int(new_width), int(new_height)))
                        # Rotate by max +/- 90°
                        elif rand_val == 1:
                            angle = round(np.random.uniform(-1, 1) * 90.0)
                            image_center = tuple(np.array(cutout.shape[1::-1]) / 2)
                            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                            cutout = cv2.warpAffine(cutout, rot_mat, cutout.shape[1::-1], flags=cv2.INTER_LINEAR)
                        # Flip image vertically
                        else:
                            cutout = cv2.flip(cutout, 1)

                # Place image region centred on square black background
                if cutout.shape[1] > cutout.shape[0]:
                    window = np.zeros((cutout.shape[1], cutout.shape[1], 3))
                    margin = int((window.shape[0] - cutout.shape[0]) / 2)
                    window[margin:(margin + cutout.shape[0]), :, :] = cutout
                else:
                    window = np.zeros((cutout.shape[0], cutout.shape[0], 3))
                    margin = int((window.shape[1] - cutout.shape[1]) / 2)
                    window[:, margin:(margin + cutout.shape[1]), :] = cutout

                window = cv2.resize(window, (self.image_width, self.image_height))
            window = np.array(window, dtype=np.float)
            window /= 255.0

            x[i] = window
            y[i] = label_vec

        return x, y