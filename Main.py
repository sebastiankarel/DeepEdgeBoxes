#!/usr/bin/env python

import tensorflow as tf
from RegionProposal import RegionProposal
from BoundingBoxRegression import BoundingBoxRegression
from Classification import Classification


import os
import numpy as np
import random
import xml.etree.ElementTree as et
import cv2
import math


def init_tf_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == "__main__":
    init_tf_gpu()

    class_labels = ['person', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bird',
                    'cat', 'cow', 'dog', 'horse', 'sheep', 'bottle', 'chair', 'diningtable', 'pottedplant',
                    'sofa', 'tvmonitor', 'none']
    classifier = Classification(224, 224, "classifier_weights.h5")

    classifier.train_model(
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        5,
        18,
        True
    )

    for fn in os.listdir("pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"):
        image = cv2.imread(os.path.join("pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages", fn))
        prediction = classifier.predict(image)
        for pred in prediction:
            label = np.argmax(pred[4])
            significance = np.amax(pred[4])
            if label != 20 and significance >= 0.5:
                print(class_labels[label])
                cv2.rectangle(image, (pred[0], pred[1]), (pred[2], pred[3]), (0, 255, 0), 1)
        cv2.imshow("image", image)
        cv2.waitKey(0)


