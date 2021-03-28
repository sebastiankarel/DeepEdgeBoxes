import keras
from DataGenerationBBoxReg import DataGenerator
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from EdgeDetection import HED
import xml.etree.ElementTree as et
from shutil import copyfile


class BoundingBoxRegression:

    def __init__(self, image_width, image_height, weight_file):
        self.image_width = image_width
        self.image_height = image_height
        self.weight_file = weight_file

    def __get_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_1", input_shape=(self.image_height, self.image_width, 1)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_2"))
        model.add(keras.layers.MaxPool2D(name="max_pool_1"))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv_2_1"))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv_2_2"))
        model.add(keras.layers.MaxPool2D(name="max_pool_2"))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_1"))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_2"))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name="conv_3_3"))
        model.add(keras.layers.MaxPool2D(name="max_pool_3"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_1"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_2"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_4_3"))
        model.add(keras.layers.MaxPool2D(name="max_pool_4"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_1"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_2"))
        model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name="conv_5_3"))
        model.add(keras.layers.MaxPool2D(name="max_pool_5"))
        model.add(keras.layers.Flatten(name="flatten"))
        model.add(keras.layers.Dense(units=128, activation='relu', name="dense_reg_1"))
        model.add(keras.layers.Dense(units=64, activation='relu', name="dense_reg_2"))
        model.add(keras.layers.Dense(units=32, activation='relu', name="dense_reg_3"))
        model.add(keras.layers.Dense(units=4, activation='sigmoid', name="out_reg"))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.MeanSquaredError())
        model.summary()
        return model

    def train_model(self, train_labels_dir, train_images_dir, test_labels_dir, test_images_dir, epochs, batch_size, load_weights):
        model = self.__get_model()
        if load_weights:
            if os.path.isfile(self.weight_file):
                model.load_weights(self.weight_file)
        training_generator = DataGenerator(
            train_images_dir,
            train_labels_dir,
            batch_size,
            self.image_width, self.image_height)
        test_generator = DataGenerator(
            test_images_dir,
            test_labels_dir,
            batch_size,
            self.image_width, self.image_height)
        model.fit_generator(generator=training_generator, validation_data=test_generator, use_multiprocessing=False, epochs=epochs)
        model.save_weights(self.weight_file, overwrite=True)

    def draw_predictions(self):
        pass