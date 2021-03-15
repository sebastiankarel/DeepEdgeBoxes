import keras
from DataGenerationBBoxReg import DataGenerator
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from EdgeDetection import HED
import xml.etree.ElementTree as et
from shutil import copyfile


train_label_dir = "data/pascalvoc07/train/annotations"
train_img_dir = "data/pascalvoc07/train/images"
train_edge_img_dir = "data/pascalvoc07/train/edge_images"

test_label_dir = "data/pascalvoc07/test/annotations"
test_img_dir = "data/pascalvoc07/test/images"
test_edge_img_dir = "data/pascalvoc07/test/edge_images"

train_split_file = "data/pascalvoc07/train_split.txt"
test_split_file = "data/pascalvoc07/test_split.txt"

img_dir = "data/pascalvoc07/images_all"
label_dir = "data/pascalvoc07/annotations_all"

train_test_split = 0.8

weight_file = "model_weights_canny.h5"

img_width = 400
img_height = 400
img_channels = 1

batch_size = 8
epochs = 25


def create_data_split_files():
    files = os.listdir(label_dir)
    np.random.shuffle(files)
    train_split = int(float(len(files)) * train_test_split)
    train_files = files[0:train_split]
    test_files = files[train_split:]
    train_out = open(train_split_file, 'x')
    for name in train_files:
        train_out.write(name + "\n")
    train_out.close()
    test_out = open(test_split_file, 'x')
    for name in test_files:
        test_out.write(name + "\n")
    test_out.close()


def create_data_from_split_files(max_objects=1):
    hed = HED()
    f_train = open(train_split_file, 'r')
    files = f_train.readlines()
    f_train.close()

    for file_name in files:
        file_name = file_name.strip()
        tree = et.parse(os.path.join(label_dir, file_name))
        root = tree.getroot()
        count = 0
        for bndbox in root.findall("./object/bndbox"):
            count += 1
        if count <= max_objects:
            img_file = file_name.split(".")[0] + ".jpg"
            if not os.path.isfile(os.path.join(train_edge_img_dir, img_file)):
                image = cv2.imread(os.path.join(img_dir, img_file))
                edge_image = hed.get_edge_image(image, image.shape[1], image.shape[0], False)
                cv2.imwrite(os.path.join(train_edge_img_dir, img_file), edge_image)
            if not os.path.isfile(os.path.join(train_img_dir, img_file)):
                copyfile(os.path.join(img_dir, img_file), os.path.join(train_img_dir, img_file))
            if not os.path.isfile(os.path.join(train_label_dir, file_name)):
                copyfile(os.path.join(label_dir, file_name), os.path.join(train_label_dir, file_name))

    f_test = open(test_split_file, 'r')
    files = f_test.readlines()
    f_test.close()

    for file_name in files:
        file_name = file_name.strip()
        tree = et.parse(os.path.join(label_dir, file_name))
        root = tree.getroot()
        count = 0
        for bndbox in root.findall("./object/bndbox"):
            count += 1
        if count <= max_objects:
            img_file = file_name.split(".")[0] + ".jpg"
            if not os.path.isfile(os.path.join(test_edge_img_dir, img_file)):
                image = cv2.imread(os.path.join(img_dir, img_file))
                edge_image = hed.get_edge_image(image, image.shape[1], image.shape[0], False)
                cv2.imwrite(os.path.join(test_edge_img_dir, img_file), edge_image)
            if not os.path.isfile(os.path.join(test_img_dir, img_file)):
                copyfile(os.path.join(img_dir, img_file), os.path.join(test_img_dir, img_file))
            if not os.path.isfile(os.path.join(test_label_dir, file_name)):
                copyfile(os.path.join(label_dir, file_name), os.path.join(test_label_dir, file_name))


def get_regression_model():
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


def train_regression_model(model, load_weights):
    if load_weights:
        if os.path.isfile(weight_file):
            model.load_weights(weight_file)
    training_generator = DataGenerator(
        train_img_dir,
        train_label_dir,
        img_width, img_height, img_channels, batch_size)
    test_generator = DataGenerator(
        test_img_dir,
        test_label_dir,
        img_width, img_height, img_channels, batch_size)
    model.fit_generator(generator=training_generator, validation_data=test_generator, use_multiprocessing=False, epochs=epochs)
    model.save_weights(weight_file, overwrite=True)


def dog(image):
    low_sigma = np.array(cv2.GaussianBlur(image, (3, 3), 0))
    high_sigma = np.array(cv2.GaussianBlur(image, (5, 5), 0))
    return low_sigma - high_sigma


def draw_prediction(model):
    if os.path.isfile(weight_file):
        model.load_weights(weight_file)
        for file_name in os.listdir(test_label_dir):
            img_file = file_name.split(".")[0] + ".jpg"
            image = cv2.imread(os.path.join(test_img_dir, img_file))
            image = cv2.resize(image, (img_width, img_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            edge_image = dog(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            edge_image = cv2.resize(edge_image, (img_width, img_height))
            edge_image = np.reshape(edge_image, (edge_image.shape[0], edge_image.shape[1], 1))
            edge_image = edge_image / 255.0
            edge_image = np.reshape(edge_image, (1, edge_image.shape[0], edge_image.shape[1], edge_image.shape[2]))
            prediction = model.predict(edge_image, 1)[0]
            xmin = prediction[0] * float(img_width)
            ymin = prediction[1] * float(img_height)
            xmax = prediction[2] * float(img_width)
            ymax = prediction[3] * float(img_height)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            plt.imshow(image)
            plt.show()
    else:
        pass