import keras
from DataGenerationClassificationVertical import DataGenerator
from DataGenerationClassificationHEDVertical import DataGenerator as DataGenHed
import os
import cv2
import numpy as np
from EdgeDetection import HED
import EdgeDetection as ed


class Classification:

    def __init__(self, image_width, image_height, weight_file, use_hed, use_multichannel, use_rgb):
        if image_width is not image_height * 2:
            raise ValueError('Image width must be 2 * image height.')
        self.image_width = image_width
        self.image_height = image_height
        self.weight_file = weight_file
        self.use_hed = use_hed
        self.use_multichannel = use_multichannel
        self.use_rgb = use_rgb
        if use_hed:
            self.hed = HED()
        else:
            self.hed = None

    def __get_model(self):
        if self.use_hed:
            channels = 1
        else:
            if self.use_multichannel or self.use_rgb:
                channels = 3
            else:
                channels = 1
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_1", input_shape=(self.image_height, self.image_width, channels)))
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
        model.add(keras.layers.Dense(units=1024, activation='relu', name="dense_class_1"))
        model.add(keras.layers.Dense(units=512, activation='relu', name="dense_class_2"))
        model.add(keras.layers.Dense(units=21, activation='softmax', name="out_class"))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy, metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        return model

    def train_model(self, train_labels_dir, train_images_dir, val_labels_dir, val_images_dir, epochs, batch_size, load_weights):
        model = self.__get_model()
        if load_weights:
            if os.path.isfile(self.weight_file):
                model.load_weights(self.weight_file)
        if self.use_hed:
            training_generator = DataGenHed(
                train_images_dir,
                train_labels_dir,
                batch_size,
                self.image_width, self.image_height, True)
            val_generator = DataGenHed(
                val_images_dir,
                val_labels_dir,
                batch_size,
                self.image_width, self.image_height, False)
        else:
            training_generator = DataGenerator(
                train_images_dir,
                train_labels_dir,
                batch_size,
                self.image_width, self.image_height, True, self.use_multichannel, self.use_rgb)
            val_generator = DataGenerator(
                val_images_dir,
                val_labels_dir,
                batch_size,
                self.image_width, self.image_height, False, self.use_multichannel, self.use_rgb)

        history = model.fit(x=training_generator, validation_data=val_generator, use_multiprocessing=False, epochs=epochs)
        model.save_weights(self.weight_file, overwrite=True)
        return history
