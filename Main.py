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


def init_tf_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == "__main__":
    init_tf_gpu()
    '''
    classifier = Classification(224, 224, "classifier_weights.h5")
    classifier.train_model(
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        45,
        18,
        False
    )
    '''
    reg_prop = RegionProposal(224, 224, "reg_prop_weights.h5", "classifier_weights.h5")
    for f in os.listdir("pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"):
        image = cv2.imread(os.path.join("pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages", f))
        if image is not None:
            prediction = reg_prop.predict(image)
            for pred in prediction:
                if pred[4][0] >= 0.2:
                    cv2.rectangle(image, (pred[0], pred[1]), (pred[2], pred[3]), (0, 255, 0), 1)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
    '''
    reg_prop.train_model(
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        18,
        18,
        False
    )
    '''
    '''
    box_reg = BoundingBoxRegression(224, 224, "bbox_weights.h5", "classifier_weights.h5")
    box_reg.train_model(
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
        "pascalvoc2007/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
        30,
        18,
        False
    )
    '''


