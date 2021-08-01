import keras
from DataGenShapePrediction import DataGenerator
from DataGenShapePredictionHED import DataGenerator as DataGenHed
import os
import cv2
import numpy as np
import EdgeDetection as ed


class ShapePrediction:

    def __init__(self, image_width, image_height, class_weights, weight_file, use_hed, use_multichannel, hed=None):
        self.image_width = image_width
        self.image_height = image_height
        self.weight_file = weight_file
        self.class_weights = class_weights
        self.use_hed = use_hed
        self.use_multichannel = use_multichannel
        if use_hed:
            self.hed = hed
        else:
            self.hed = None

    def __get_model(self, load_weights=True):
        if self.use_hed:
            channels = 1
        else:
            if self.use_multichannel:
                channels = 3
            else:
                channels = 1
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_1", input_shape=(self.image_height, self.image_width, channels), trainable=False))
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
        model.add(keras.layers.Flatten(name="flatten"))
        model.add(keras.layers.Dense(units=1024, activation='relu', name="dense_shape_1"))
        model.add(keras.layers.Dense(units=512, activation='relu', name="dense_shape_2"))
        model.add(keras.layers.Dense(units=9, activation='softmax', name="out_shape"))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy, metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        if load_weights:
            if os.path.isfile(self.weight_file):
                model.load_weights(self.weight_file)
        return model

    def train_model(self, train_labels_dir, train_images_dir, val_labels_dir, val_images_dir, epochs, batch_size, load_weights):
        model = self.__get_model(load_weights)
        if os.path.isfile(self.class_weights):
            model.load_weights(self.class_weights, by_name=True)
        else:
            raise Exception("Classifier weights not found")
        if self.use_hed:
            training_generator = DataGenHed(
                train_images_dir,
                train_labels_dir,
                batch_size,
                self.image_width, self.image_height)
            val_generator = DataGenHed(
                val_images_dir,
                val_labels_dir,
                batch_size,
                self.image_width, self.image_height)
        else:
            training_generator = DataGenerator(
                train_images_dir,
                train_labels_dir,
                batch_size,
                self.image_width, self.image_height, self.use_multichannel)
            val_generator = DataGenerator(
                val_images_dir,
                val_labels_dir,
                batch_size,
                self.image_width, self.image_height, self.use_multichannel)

        history = model.fit(x=training_generator, validation_data=val_generator, use_multiprocessing=False, epochs=epochs)
        model.save_weights(self.weight_file, overwrite=True)
        return history

    def predict(self, original_image, predictions):
        model = self.__get_model(True)
        result = []
        for prediction in predictions:
            image = original_image[prediction[1]:prediction[3], prediction[0]:prediction[2], :]
            orig_width = image.shape[1]
            orig_height = image.shape[0]

            if self.use_hed:
                edge_image = self.hed.get_edge_image(image, orig_width, orig_height, normalized=False)
                edge_image = np.reshape(edge_image, (edge_image.shape[0], edge_image.shape[1], 1))
            else:
                edge_image = ed.auto_canny(image, self.use_multichannel)
                if not self.use_multichannel:
                    edge_image = np.reshape(edge_image, (edge_image.shape[0], edge_image.shape[1], 1))

            # Zero padding to make square
            if edge_image.shape[0] > edge_image.shape[1]:
                x_padding = edge_image.shape[0] - edge_image.shape[1]
                x_padding_start = round(x_padding / 2.0)
                x_end = x_padding_start + edge_image.shape[1]
                new_edge_image = np.zeros((edge_image.shape[0], edge_image.shape[1] + x_padding, edge_image.shape[2]))
                new_edge_image[:, x_padding_start:x_end, :] = edge_image[:, :, :]
                edge_image = new_edge_image
            elif edge_image.shape[1] > edge_image.shape[0]:
                y_padding = edge_image.shape[1] - edge_image.shape[0]
                y_padding_top = round(y_padding / 2.0)
                y_end = y_padding_top + edge_image.shape[0]
                new_edge_image = np.zeros((edge_image.shape[0] + y_padding, edge_image.shape[1], edge_image.shape[2]))
                new_edge_image[y_padding_top:y_end, :, :] = edge_image[:, :, :]
                edge_image = new_edge_image

            resized_image = np.array(cv2.resize(edge_image, (self.image_width, self.image_height)), dtype=np.float)
            if not self.use_multichannel:
                resized_image = np.reshape(resized_image, (1, resized_image.shape[0], resized_image.shape[1], 1))
            else:
                resized_image = np.reshape(resized_image, (1, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2]))

            result.append(model.predict(resized_image, 1)[0])
        return result
