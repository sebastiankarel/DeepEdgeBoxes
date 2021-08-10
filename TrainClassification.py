import tensorflow as tf
from Classification import Classification
import sys


def init_tf_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == "__main__":
    init_tf_gpu()

    edge_type = "single_canny"
    batch_size = 18
    epochs = 50
    for arg in sys.argv:
        split = arg.split("=")
        if len(split) == 2:
            if split[0] == "edge_type":
                if split[1] == "single_canny" or split[1] == "multi_canny" or split[1] == "hed" or split[1] == "rgb_canny":
                    edge_type = split[1]
                else:
                    print("Unknown edge type {}. Using default single_canny".format(split[1]))
            elif split[0] == "batch_size":
                batch_size = int(split[1])
            elif split[0] == "epochs":
                epochs = int(split[1])
            else:
                print("Unknown argument {}. Ignoring it.".format(split[0]))

    print("Read train_configs.txt")
    file = open("train_configs.txt", "r")
    lines = file.readlines()

    train_images_dir = ""
    train_labels_dir = ""
    val_images_dir = ""
    val_labels_dir = ""

    if edge_type == "multi_canny":
        weight_file = "classifier_weights_multi.h5"
    elif edge_type == "hed":
        weight_file = "classifier_weights_hed.h5"
    elif edge_type == "rgb_canny":
        weight_file = "classifier_weights_rgb.h5"
    else:
        weight_file = "classifier_weights.h5"

    use_hed = edge_type == "hed"
    use_multi = edge_type == "multi_canny"
    use_rgb = edge_type == "rgb_canny"

    for line in lines:
        split = line.split("=")
        if len(split) == 2:
            if edge_type == "single_canny" or edge_type == "multi_canny" or edge_type == "rgb_canny":
                if split[0] == "train_images_dir":
                    train_images_dir = split[1]
                elif split[0] == "train_labels_dir":
                    train_labels_dir = split[1]
                elif split[0] == "val_images_dir":
                    val_images_dir = split[1]
                elif split[0] == "val_labels_dir":
                    val_labels_dir = split[1]
            elif edge_type == "hed":
                if split[0] == "hed_train_images_dir":
                    train_images_dir = split[1]
                elif split[0] == "hed_train_labels_dir":
                    train_labels_dir = split[1]
                elif split[0] == "hed_val_images_dir":
                    val_images_dir = split[1]
                elif split[0] == "hed_val_labels_dir":
                    val_labels_dir = split[1]

    classifier = Classification(224, 224, weight_file=weight_file, use_hed=use_hed, use_multichannel=use_multi, use_rgb=use_rgb)
    print("Starting training...")
    history = classifier.train_model(
        train_labels_dir.strip(),
        train_images_dir.strip(),
        val_labels_dir.strip(),
        val_images_dir.strip(),
        epochs,
        batch_size,
        False
    )
