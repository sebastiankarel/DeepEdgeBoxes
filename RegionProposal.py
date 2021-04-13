import keras
from DataGenerationProposal import DataGenerator as PropGen
import cv2
import os
import numpy as np


class RegionProposal:

    def __init__(self, window_width, window_height, weight_file, classifier_weights):
        self.window_width = window_width
        self.window_height = window_height
        self.weight_file = weight_file
        self.classfifier_weights = classifier_weights

    def __get_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_1", trainable=False,
                                      input_shape=(self.window_height, self.window_width, 1)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_2", trainable=False))
        model.add(keras.layers.MaxPool2D(name="max_pool_1"))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv_2_1", trainable=False))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv_2_2", trainable=False))
        model.add(keras.layers.MaxPool2D(name="max_pool_2"))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_1", trainable=False))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_2", trainable=False))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_3", trainable=False))
        model.add(keras.layers.MaxPool2D(name="max_pool_3"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_1", trainable=False))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_2", trainable=False))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_3", trainable=False))
        model.add(keras.layers.MaxPool2D(name="max_pool_4"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_1", trainable=False))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_2", trainable=False))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_3", trainable=False))
        model.add(keras.layers.MaxPool2D(name="max_pool_5"))
        model.add(keras.layers.Flatten(name="flatten"))
        model.add(keras.layers.Dense(units=128, activation='relu', name="dense_dis_1"))
        model.add(keras.layers.Dense(units=64, activation='relu', name="dense_dis_2"))
        model.add(keras.layers.Dense(units=32, activation='relu', name="dense_dis_3"))
        model.add(keras.layers.Dense(units=1, activation='sigmoid', name="out_dis"))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.MeanSquaredError())
        model.load_weights(self.classfifier_weights, by_name=True)
        model.summary()
        return model

    def train_model(self, train_labels_dir, train_images_dir, test_labels_dir, test_images_dir, epochs, batch_size, load_weights):
        model = self.__get_model()
        if load_weights:
            if os.path.isfile(self.weight_file):
                model.load_weights(self.weight_file)
        training_generator = PropGen(
            train_images_dir,
            train_labels_dir,
            batch_size,
            self.window_width,
            self.window_height)
        test_generator = PropGen(
            test_images_dir,
            test_labels_dir,
            batch_size,
            self.window_width,
            self.window_height)
        model.fit(x=training_generator, validation_data=test_generator, use_multiprocessing=False, epochs=epochs)
        model.save_weights(self.weight_file, overwrite=True)

    def predict(self, image):
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
        edge_image = np.dstack((no_sigma, low_sigma, high_sigma))

        # Run exhaustive sliding window
        result = []
        model = self.__get_model()
        model.load_weights(self.weight_file)
        window_width = self.window_width
        window_height = self.window_height
        scales = range(2, 6)
        overlap = 0.5
        for scale in scales:
            image_width = scale * window_width
            image_height = scale * window_height
            steps = int(((scale / overlap) - 1))
            for x in range(0, steps):
                x_offset = x * (window_width * overlap)
                for y in range(0, steps):
                    y_offset = y * (window_height * overlap)
                    resized_image = np.array(cv2.resize(edge_image, (image_width, image_height)), dtype=np.float)
                    xmin = int(x_offset)
                    xmax = int(x_offset + window_width)
                    ymin = int(y_offset)
                    ymax = int(y_offset + window_height)
                    window = resized_image[ymin:ymax, xmin:xmax]
                    window = np.reshape(window, (1, window.shape[0], window.shape[1], window.shape[2]))
                    window = window / 255.0
                    prediction = model.predict(window, 1)[0]
                    true_xmin = round(x_offset / scale)
                    true_ymin = round(y_offset / scale)
                    true_xmax = true_xmin + round(window_width / scale)
                    true_ymax = true_ymin + round(window_height / scale)
                    window_result = [true_xmin, true_ymin, true_xmax, true_ymax, prediction]
                    result.append(window_result)
        return result
