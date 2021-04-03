#!/usr/bin/env python

import tensorflow as tf
from RegionProposal import RegionProposal
from BoundingBoxRegression import BoundingBoxRegression
from Classification import Classification


import os
import numpy as np
import cv2
import xml.etree.ElementTree as et
import random


def init_tf_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


if __name__ == "__main__":
    init_tf_gpu()

    classifier = Classification(224, 224, "classifier_weights.h5")
    classifier.train_model(
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        20,
        16,
        False
    )

    box_reg = BoundingBoxRegression(224, 224, "bbox_weights.h5", "classifier_weights.h5")
    box_reg.train_model(
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        30,
        16,
        False
    )

    #reg_prop = RegionProposal(400, 400, "reg_prop_weights.h5")
    #reg_prop.train_model("data/train/labels/", "data/train/images", "data/test/labels", "data/test/images", 25, 8, False)


