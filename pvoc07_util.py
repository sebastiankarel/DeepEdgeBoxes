#!/usr/bin/python

from EdgeDetection import HED
import os
import cv2
from shutil import copyfile
import random


class PascalVoc2007DatasetGenerator:

    def __init__(self, train_labels_dir, train_images_dir, test_labels_dir, test_images_dir, hed_out_dir, original_out_dir):
        self.train_labels_dir = train_labels_dir
        self.train_images_dir = train_images_dir
        self.test_labels_dir = test_labels_dir
        self.test_images_dir = test_images_dir
        self.hed_out_dir = hed_out_dir
        self.original_out_dir = original_out_dir
        self.hed = HED()

    def create_split_files(self):
        print("Creating split files...")
        train_split_file = open("train_split.txt", "x")
        val_split_file = open("val_split.txt", "x")
        test_split_file = open("test_split.txt", "x")

        train_file_names = []
        for file_name in os.listdir(self.train_labels_dir):
            image_file_name = "{}{}".format(file_name.split(".")[0], ".jpg")
            if os.path.isfile(os.path.join(self.train_images_dir, image_file_name)):
                train_file_names.append("{}{}".format(file_name.split(".")[0], "\n"))

        train_size = len(train_file_names)
        val_split = round(train_size / 10)
        val_file_names = random.sample(train_file_names, val_split)
        for val_file in val_file_names:
            train_file_names.remove(val_file)
        
        test_file_names = []
        for file_name in os.listdir(self.test_labels_dir):
            image_file_name = "{}{}".format(file_name.split(".")[0], ".jpg")
            if os.path.isfile(os.path.join(self.test_images_dir, image_file_name)):
                test_file_names.append("{}{}".format(file_name.split(".")[0], "\n"))

        train_split_file.writelines(train_file_names)
        val_split_file.writelines(val_file_names)
        test_split_file.writelines(test_file_names)

        train_split_file.close()
        val_split_file.close()
        test_split_file.close()
        print("Split files created.")

    def create_original_dataset(self):
        self.create_original_train_dataset()
        self.create_original_val_dataset()
        self.create_original_test_dataset()

    def create_hed_dataset(self):
        self.create_hed_train_dataset()
        self.create_hed_val_dataset()
        self.create_hed_test_dataset()

    def create_hed_train_dataset(self):
        train_dir_out = os.path.join(self.hed_out_dir, "train")
        if not os.path.exists(train_dir_out):
            os.makedirs(train_dir_out)
        images_dir_out = os.path.join(train_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(train_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        file_names_file = open("train_split.txt", "r")
        file_names = file_names_file.readlines()
        print("Creating hed train set with {} files".format(len(file_names)))
        for file_name in file_names:
            file_name = file_name.strip()
            image_file_name = "{}{}".format(file_name, ".jpg")
            label_file_name = "{}{}".format(file_name, ".xml")
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
        file_names_file.close()
        print("Hed train set created.")

    def create_hed_val_dataset(self):
        val_dir_out = os.path.join(self.hed_out_dir, "val")
        if not os.path.exists(val_dir_out):
            os.makedirs(val_dir_out)
        images_dir_out = os.path.join(val_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(val_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        file_names_file = open("val_split.txt", "r")
        file_names = file_names_file.readlines()
        print("Creating hed val set with {} files".format(len(file_names)))
        for file_name in file_names:
            file_name = file_name.strip()
            image_file_name = "{}{}".format(file_name, ".jpg")
            label_file_name = "{}{}".format(file_name, ".xml")
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
        print("Hed val set created.")

    def create_hed_test_dataset(self):
        test_dir_out = os.path.join(self.hed_out_dir, "test")
        if not os.path.exists(test_dir_out):
            os.makedirs(test_dir_out)
        images_dir_out = os.path.join(test_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(test_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        file_names_file = open("test_split.txt", "r")
        file_names = file_names_file.readlines()
        print("Creating hed test set with {} files".format(len(file_names)))
        for file_name in file_names:
            file_name = file_name.strip()
            image_file_name = "{}{}".format(file_name, ".jpg")
            label_file_name = "{}{}".format(file_name, ".xml")
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
        print("Hed test set created.")

    def create_original_train_dataset(self):
        train_dir_out = os.path.join(self.original_out_dir, "train")
        if not os.path.exists(train_dir_out):
            os.makedirs(train_dir_out)
        images_dir_out = os.path.join(train_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(train_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        file_names_file = open("train_split.txt", "r")
        file_names = file_names_file.readlines()
        print("Creating original train set with {} files".format(len(file_names)))
        for file_name in file_names:
            file_name = file_name.strip()
            image_file_name = "{}{}".format(file_name, ".jpg")
            label_file_name = "{}{}".format(file_name, ".xml")
            image_out_file = os.path.join(images_dir_out, image_file_name)
            label_out_file = os.path.join(labels_dir_out, label_file_name)
            if not os.path.isfile(image_out_file):
                image_in_file = os.path.join(self.train_images_dir, image_file_name)
                copyfile(image_in_file, image_out_file)
            if not os.path.isfile(label_out_file):
                label_in_file = os.path.join(self.train_labels_dir, label_file_name)
                copyfile(label_in_file, label_out_file)
        print("Original train set created.")

    def create_original_val_dataset(self):
        val_dir_out = os.path.join(self.original_out_dir, "val")
        if not os.path.exists(val_dir_out):
            os.makedirs(val_dir_out)
        images_dir_out = os.path.join(val_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(val_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        file_names_file = open("val_split.txt", "r")
        file_names = file_names_file.readlines()
        print("Creating original val set with {} files".format(len(file_names)))
        for file_name in file_names:
            file_name = file_name.strip()
            image_file_name = "{}{}".format(file_name, ".jpg")
            label_file_name = "{}{}".format(file_name, ".xml")
            image_out_file = os.path.join(images_dir_out, image_file_name)
            label_out_file = os.path.join(labels_dir_out, label_file_name)
            if not os.path.isfile(image_out_file):
                image_in_file = os.path.join(self.train_images_dir, image_file_name)
                copyfile(image_in_file, image_out_file)
            if not os.path.isfile(label_out_file):
                label_in_file = os.path.join(self.train_labels_dir, label_file_name)
                copyfile(label_in_file, label_out_file)
        print("Original val set created.")

    def create_original_test_dataset(self):
        test_dir_out = os.path.join(self.original_out_dir, "test")
        if not os.path.exists(test_dir_out):
            os.makedirs(test_dir_out)
        images_dir_out = os.path.join(test_dir_out, "images")
        if not os.path.exists(images_dir_out):
            os.makedirs(images_dir_out)
        labels_dir_out = os.path.join(test_dir_out, "labels")
        if not os.path.exists(labels_dir_out):
            os.makedirs(labels_dir_out)
        file_names_file = open("test_split.txt", "r")
        file_names = file_names_file.readlines()
        print("Creating original test set with {} files".format(len(file_names)))
        for file_name in file_names:
            file_name = file_name.strip()
            image_file_name = "{}{}".format(file_name, ".jpg")
            label_file_name = "{}{}".format(file_name, ".xml")
            image_out_file = os.path.join(images_dir_out, image_file_name)
            label_out_file = os.path.join(labels_dir_out, label_file_name)
            if not os.path.isfile(image_out_file):
                image_in_file = os.path.join(self.test_images_dir, image_file_name)
                copyfile(image_in_file, image_out_file)
            if not os.path.isfile(label_out_file):
                label_in_file = os.path.join(self.test_labels_dir, label_file_name)
                copyfile(label_in_file, label_out_file)
        print("Original test set created.")


if __name__ == "__main__":
    print("Reading pvoc_paths.txt...")
    train_images = ""
    train_labels = ""
    test_images = ""
    test_labels = ""
    hed_out_dir = ""
    original_out_dir = ""

    paths_file = open("pvoc07_paths.txt", "r")
    lines = paths_file.readlines()
    for line in lines:
        line = line.strip()
        split_line = line.split(" ")
        tag = split_line[0]
        value = split_line[1]
        if tag == "train_images":
            train_images = value
        elif tag == "train_labels":
            train_labels = value
        elif tag == "test_images":
            test_images = value
        elif tag == "test_labels":
            test_labels = value
        elif tag == "hed_out_dir":
            hed_out_dir = value
        elif tag == "original_out_dir":
            original_out_dir = value
    paths_file.close()

    gen = PascalVoc2007DatasetGenerator(train_labels, train_images, test_labels, test_images, hed_out_dir, original_out_dir)
    gen.create_split_files()
    gen.create_hed_dataset()
    gen.create_original_dataset()

    os.remove("train_split.txt")
    os.remove("val_split.txt")
    os.remove("test_split.txt")

    print("Done.")
