#!/usr/bin/env python

import os
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def plot_distortion_for_num_clusters(data):
    distortion = []
    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        distortion.append(kmeans.inertia_)
    plt.plot(range(1, 20), distortion)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("kmeans_elbow_method.png")


def load_train_bboxes(labels_dir):
    bboxes = []
    for label_file in os.listdir(labels_dir):
        tree = et.parse(os.path.join(labels_dir, label_file))
        root = tree.getroot()
        for obj in root.findall("./object"):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            width = xmax - xmin
            height = ymax - ymin
            bboxes.append((width, height))
    return np.array(bboxes)


def plot_clusters(data, cluster_centers):
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red')
    plt.savefig("bbox_clusters.png")


if __name__ == "__main__":
    data = load_train_bboxes("data/original/train/labels")
    # plot_distortion_for_num_clusters(data)

    kmeans = KMeans(n_clusters=12, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred = kmeans.fit_predict(data)
    # plot_clusters(data, kmeans.cluster_centers_)

    f = open("cluster_centers.txt", "x")
    for i, cluster_center in enumerate(kmeans.cluster_centers_):
        f.write("Cluster {}: width={}, height={}\n".format(i, cluster_center[0], cluster_center[1]))
    f.close()