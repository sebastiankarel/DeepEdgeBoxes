#!/usr/bin/env python

import tensorflow as tf
from RegionProposal import RegionProposal


def init_tf_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == "__main__":
    init_tf_gpu()

    reg_prop = RegionProposal(400, 400, "reg_prop_weights.h5")
    reg_prop.train_model("data/train/labels/", "data/train/images", "data/test/labels", "data/test/images", 25, 8, False)

