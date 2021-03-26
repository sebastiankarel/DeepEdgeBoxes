import keras
from DataGenerationProposal import DataGenerator as PropGen
import cv2
import os
import numpy as np


class RegionProposal:

    def __init__(self, window_width, window_height, weight_file):
        self.window_width = window_width
        self.window_height = window_height
        self.weight_file = weight_file

    def __get_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_1", trainable=True,
                                      input_shape=(self.window_height, self.window_width, 1)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_2", trainable=True))
        model.add(keras.layers.MaxPool2D(name="max_pool_1"))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv_2_1", trainable=True))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv_2_2", trainable=True))
        model.add(keras.layers.MaxPool2D(name="max_pool_2"))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_1", trainable=True))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_2", trainable=True))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_3", trainable=True))
        model.add(keras.layers.MaxPool2D(name="max_pool_3"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_1", trainable=True))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_2", trainable=True))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_3", trainable=True))
        model.add(keras.layers.MaxPool2D(name="max_pool_4"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_1", trainable=True))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_2", trainable=True))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_3", trainable=True))
        model.add(keras.layers.MaxPool2D(name="max_pool_5"))
        model.add(keras.layers.Flatten(name="flatten"))
        model.add(keras.layers.Dense(units=128, activation='relu', name="dense_dis_1"))
        model.add(keras.layers.Dense(units=64, activation='relu', name="dense_dis_2"))
        model.add(keras.layers.Dense(units=32, activation='relu', name="dense_dis_3"))
        model.add(keras.layers.Dense(units=1, activation='sigmoid', name="out_dis"))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.MeanSquaredError())
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

    def predict(self, edge_image):
        result = []
        if os.path.isfile(self.weight_file):
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
                        edge_image = np.array(cv2.resize(edge_image, (image_width, image_height)), dtype=np.float)
                        xmin = int(x_offset)
                        xmax = int(x_offset + window_width)
                        ymin = int(y_offset)
                        ymax = int(y_offset + window_height)
                        edge_image = edge_image[ymin:ymax, xmin:xmax]
                        edge_image = np.reshape(edge_image, (edge_image.shape[0], edge_image.shape[1], 1))
                        edge_image = np.reshape(edge_image, (1, edge_image.shape[0], edge_image.shape[1], edge_image.shape[2]))
                        edge_image = edge_image / 255.0
                        prediction = model.predict(edge_image, 1)[0]
                        # TODO normalise for scale
                        window_result = [x_offset, y_offset, x_offset + window_width, y_offset + window_height, prediction]
                        result.append(window_result)
        return result
