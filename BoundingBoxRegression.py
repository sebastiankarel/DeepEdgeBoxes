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

    def __init__(self, dataset_dir):
        self.train_img_dir = os.path.join(dataset_dir, "train/images")
        self.train_label_dir = os.path.join(dataset_dir, "train/labels")
        self.test_img_dir = os.path.join(dataset_dir, "test/images")
        self.test_label_dir = os.path.join(dataset_dir, "test/labels")
        self.img_width = 400
        self.img_height = 400
        self.batch_size = 8
        self.epochs = 25
        self.weight_file = "bbox_reg_weights.h5"

    @staticmethod
    def __get_model():
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_1", input_shape=(img_height, img_width, img_channels)))
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

    def train_model(self, model):
        training_generator = DataGenerator(
            self.train_img_dir,
            self.train_label_dir,
            self.img_width, self.img_height, self.batch_size)
        test_generator = DataGenerator(
            self.test_img_dir,
            self.test_label_dir,
            self.img_width, self.img_height, self.batch_size)
        model.fit_generator(generator=training_generator, validation_data=test_generator, use_multiprocessing=False, epochs=epochs)
        model.save_weights(self.weight_file, overwrite=True)

    def draw_predictions(self):
        if os.path.isfile(self.weight_file):
            model = self.__get_model()
            model.load_weights(self.weight_file)
            for file_name in os.listdir(self.test_label_dir):
                img_file = file_name.split(".")[0] + ".jpg"
                image = cv2.imread(os.path.join(self.test_img_dir, img_file))
                image = cv2.resize(image, (self.img_width, self.img_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.reshape(image, (image.shape[0], image.shape[1], 1))
                image = image / 255.0
                image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
                prediction = model.predict(image, 1)[0]
                xmin = prediction[0] * float(self.img_width)
                ymin = prediction[1] * float(self.img_height)
                xmax = prediction[2] * float(self.img_width)
                ymax = prediction[3] * float(self.img_height)
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                plt.imshow(image)
                plt.show()
        else:
            pass