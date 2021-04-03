#!/usr/bin/python

from EdgeDetection import HED
import os
import cv2
from shutil import copyfile
import argparse


class PascalVoc2007DatasetGenerator:

    def __init__(self, train_labels_dir, train_images_dir, test_labels_dir, test_images_dir, out_dir):
        self.train_labels_dir = train_labels_dir
        self.train_images_dir = train_images_dir
        self.test_labels_dir = test_labels_dir
        self.test_images_dir = test_images_dir
        self.out_dir = out_dir
        self.hed = HED()

    def create_dataset(self):
        self.create_train_dataset()
        self.create_test_dataset()

    def create_train_dataset(self):
        train_dir_out = os.path.join(self.out_dir, "train")
        if not os.path.exists(train_dir_out):
            os.makedirs(train_dir_out)
        images_dir_out = os.path.join(train_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(train_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        label_file_names = os.listdir(self.train_labels_dir)
        for label_file_name in label_file_names:
            file_name = label_file_name.split(".")[0]
            image_file_name = "{}{}".format(file_name, ".jpg")
            image = cv2.imread(os.path.join(self.train_images_dir, image_file_name))
            if image is not None:
                image_out_file = os.path.join(images_dir_out, image_file_name)
                label_out_file = os.path.join(labels_dir_out, label_file_name)
                if not os.path.isfile(image_out_file):
                    width = image.shape[1]
                    height = image.shape[0]
                    edge_image = self.hed.get_edge_image(image, width, height, normalized=False)
                    cv2.imwrite(image_out_file, edge_image)
                if not os.path.isfile(label_out_file):
                    label_in_file = os.path.join(self.train_labels_dir, label_file_name)
                    copyfile(label_in_file, label_out_file)

    def create_test_dataset(self):
        test_dir_out = os.path.join(self.out_dir, "test")
        if not os.path.exists(test_dir_out):
            os.makedirs(test_dir_out)
        images_dir_out = os.path.join(test_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(test_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        label_file_names = os.listdir(self.test_labels_dir)
        for label_file_name in label_file_names:
            file_name = label_file_name.split(".")[0]
            image_file_name = "{}{}".format(file_name, ".jpg")
            image = cv2.imread(os.path.join(self.test_images_dir, image_file_name))
            if image is not None:
                image_out_file = os.path.join(images_dir_out, image_file_name)
                label_out_file = os.path.join(labels_dir_out, label_file_name)
                if not os.path.isfile(image_out_file):
                    width = image.shape[1]
                    height = image.shape[0]
                    edge_image = self.hed.get_edge_image(image, width, height, normalized=False)
                    cv2.imwrite(image_out_file, edge_image)
                if not os.path.isfile(label_out_file):
                    label_in_file = os.path.join(self.test_labels_dir, label_file_name)
                    copyfile(label_in_file, label_out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training and test datasets.')
    parser.add_argument('-train_labels_dir', type=str, help='The directory of the Pascal VOC 2007 training labels')
    parser.add_argument('-train_images_dir', type=str, help='The directory of the Pascal VOC 2007 training images')
    parser.add_argument('-test_labels_dir', type=str, help='The directory of the Pascal VOC 2007 test labels')
    parser.add_argument('-test_images_dir', type=str, help='The directory of the Pascal VOC 2007 test images')
    parser.add_argument('-out_dir', metavar='out_dir', type=str, help='The output directory for the processed dataset')
    args = parser.parse_args()

    gen = PascalVoc2007DatasetGenerator(args.train_labels_dir, args.train_images_dir, args.test_labels_dir, args.test_images_dir, args.out_dir)
    gen.create_dataset()