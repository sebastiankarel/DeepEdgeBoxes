import keras
from DataGenerationProposal import DataGenerator as PropGen
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from EdgeDetection import HED
import xml.etree.ElementTree as et
from shutil import copyfile

train_label_dir = "data/pascalvoc07/train_2/annotations"
train_img_dir = "data/pascalvoc07/train_2/images"
train_edge_img_dir = "data/pascalvoc07/train_2/edge_images"

test_label_dir = "data/pascalvoc07/test_2/annotations"
test_img_dir = "data/pascalvoc07/test_2/images"
test_edge_img_dir = "data/pascalvoc07/test_2/edge_images"

train_split_file = "data/pascalvoc07/train_split_2.txt"
test_split_file = "data/pascalvoc07/test_split_2.txt"

img_dir = "data/pascalvoc07/images_all"
label_dir = "data/pascalvoc07/annotations_all"

train_test_split = 0.8

weight_file = "model_weights_dis.h5"

img_width = 400
img_height = 400
img_channels = 1

batch_size = 8
epochs = 25


class RegionProposal:

    def create_data_split_files(self):
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

    def create_data_from_split_files(self):
        hed = HED()
        f_train = open(train_split_file, 'r')
        files = f_train.readlines()
        f_train.close()

        for file_name in files:
            file_name = file_name.strip()
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
            img_file = file_name.split(".")[0] + ".jpg"
            if not os.path.isfile(os.path.join(test_edge_img_dir, img_file)):
                image = cv2.imread(os.path.join(img_dir, img_file))
                edge_image = hed.get_edge_image(image, image.shape[1], image.shape[0], False)
                cv2.imwrite(os.path.join(test_edge_img_dir, img_file), edge_image)
            if not os.path.isfile(os.path.join(test_img_dir, img_file)):
                copyfile(os.path.join(img_dir, img_file), os.path.join(test_img_dir, img_file))
            if not os.path.isfile(os.path.join(test_label_dir, file_name)):
                copyfile(os.path.join(label_dir, file_name), os.path.join(test_label_dir, file_name))

    def get_discriminator_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv_1_1", trainable=False,
                                      input_shape=(img_height, img_width, img_channels)))
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
        model.summary()
        model.load_weights("model_weights.h5", by_name=True)
        return model

    def train_discriminator_model(self, model, load_weights):
        if load_weights:
            if os.path.isfile(weight_file):
                model.load_weights(weight_file)
        training_generator = PropGen(
            train_edge_img_dir,
            train_label_dir,
            batch_size)
        test_generator = PropGen(
            test_edge_img_dir,
            test_label_dir,
            batch_size)
        model.fit_generator(generator=training_generator, validation_data=test_generator, use_multiprocessing=False,
                            epochs=epochs)
        model.save_weights(weight_file, overwrite=True)

    def draw_prediction(self, model):
        model.load_weights(weight_file)
        window_width = img_width
        window_height = img_height
        for i, f in enumerate(os.listdir(test_label_dir)):
            f = f.split(".")[0]
            original_image = cv2.imread(os.path.join(test_img_dir, f + ".jpg"))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            orig_width = int(original_image.shape[1])
            orig_height = int(original_image.shape[0])
            scales = range(2, 6)
            overlap = 0.5
            for scale in scales:
                image_width = scale * window_width
                image_height = scale * window_height
                original_image = cv2.resize(original_image, (image_width, image_height))
                steps = int(((scale / overlap) - 1))
                for x in range(0, steps):
                    x_offset = x * (window_width * overlap)
                    for y in range(0, steps):
                        y_offset = y * (window_height * overlap)

                        edge_image = cv2.imread(os.path.join(test_edge_img_dir, f + ".jpg"))
                        edge_image = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
                        edge_image = np.array(cv2.resize(edge_image, (image_width, image_height)), dtype=np.float)
                        xmin = int(x_offset)
                        xmax = int(x_offset + window_width)
                        ymin = int(y_offset)
                        ymax = int(y_offset + window_height)
                        edge_image = edge_image[ymin:ymax, xmin:xmax]
                        edge_image = np.reshape(edge_image, (edge_image.shape[0], edge_image.shape[1], 1))
                        edge_image = np.reshape(edge_image,
                                                (1, edge_image.shape[0], edge_image.shape[1], edge_image.shape[2]))
                        edge_image = edge_image / 255.0
                        prediction = model.predict(edge_image, 1)[0]

                        x_center = int(x_offset + (window_width / 2))
                        y_center = int(y_offset + (window_height / 2))
                        radius = scale * 10
                        thickness = scale * 4
                        if prediction < 0.5:
                            color = (0, 255, 0)
                        else:
                            color = (255, 0, 0)

                        if prediction > 0.5:
                            cv2.circle(original_image, (x_center, y_center), radius, color, thickness)

            original_image = cv2.resize(original_image, (orig_width, orig_height))
            plt.imshow(original_image)
            plt.show()
