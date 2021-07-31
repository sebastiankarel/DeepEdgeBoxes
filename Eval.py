import tensorflow as tf
from BinaryClassification import Classification
import sys
import os
import cv2
import numpy as np
import xml.etree.ElementTree as et


def init_tf_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


def compute_iou(ground_truth, prediction):
    x_overlap = max(0, min(ground_truth[2], prediction[2]) - max(ground_truth[0], prediction[0]))
    y_overlap = max(0, min(ground_truth[3], prediction[3]) - max(ground_truth[1], prediction[1]))
    intersection = float(x_overlap * y_overlap)
    gt_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
    pred_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    union = float(gt_area + pred_area) - intersection
    return intersection / union


def get_num_matching_predictions(ground_truths, predictions, class_threshold=0.5, iou_threshold=0.5):
    best_predictions = []
    for ground_truth in ground_truths:
        best_prediction = (None, 0.0)
        for prediction in predictions:
            if prediction[4] > class_threshold:
                iou = compute_iou(ground_truth, prediction)
                if iou >= iou_threshold:
                    if iou > best_prediction[1]:
                        best_prediction = (prediction, iou)
        best_predictions.append(best_prediction[0])
    num_predicted = 0
    num_missed = 0
    for best in best_predictions:
        if best is None:
            num_missed += 1
        else:
            num_predicted += 1
    return num_predicted, num_missed


def read_label_file(file_name):
    bboxes = []
    tree = et.parse(file_name)
    root = tree.getroot()
    for obj in root.findall("./object"):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return np.array(bboxes)


if __name__ == "__main__":
    init_tf_gpu()

    edge_type = "single_canny"
    for arg in sys.argv:
        split = arg.split("=")
        if len(split) == 2:
            if split[0] == "edge_type":
                if split[1] == "single_canny" or split[1] == "multi_canny" or split[1] == "hed":
                    edge_type = split[1]
                else:
                    print("Unknown edge type {}. Using default single_canny".format(split[1]))
            else:
                print("Unknown argument {}. Ignoring it.".format(split[0]))

    print("Read eval_configs.txt")
    file = open("eval_configs.txt", "r")
    lines = file.readlines()

    test_images_dir = ""
    test_labels_dir = ""

    if edge_type == "multi_canny":
        weight_file = "bin_classifier_weights_multi.h5"
        class_weight_file = "classifier_weights_multi.h5"
        reg_weight_file = "shape_classifier_weights_multi.h5.h5"
    elif edge_type == "hed":
        weight_file = "bin_classifier_weights_hed.h5"
        class_weight_file = "classifier_weights_hed.h5"
        reg_weight_file = "shape_classifier_weights_hed.h5.h5"
    else:
        weight_file = "bin_classifier_weights.h5"
        class_weight_file = "classifier_weights.h5"
        reg_weight_file = "shape_classifier_weights.h5"

    use_hed = edge_type == "hed"
    use_multi = edge_type == "multi_canny"

    for line in lines:
        split = line.split("=")
        if len(split) == 2:
            if edge_type == "single_canny" or edge_type == "multi_canny":
                if split[0] == "test_images_dir":
                    test_images_dir = split[1]
                elif split[0] == "test_labels_dir":
                    test_labels_dir = split[1]
            elif edge_type == "hed":
                if split[0] == "hed_test_images_dir":
                    test_images_dir = split[1]
                elif split[0] == "hed_test_labels_dir":
                    test_labels_dir = split[1]

    classifier = Classification(224, 224, class_weights=class_weight_file, weight_file=weight_file, use_hed=use_hed, use_multichannel=use_multi)
    print("Starting evaluation")
    test_images_dir = test_images_dir.strip()
    test_labels_dir = test_labels_dir.strip()
    labels = os.listdir(test_labels_dir)

    true_positives = 0
    false_negatives = 0
    for i, label_file_name in enumerate(labels):
        image_file_name = label_file_name.split(".")[0] + ".jpg"
        image = cv2.imread(os.path.join(test_images_dir, image_file_name))
        if image is not None:
            print("Evaluating image {} of {}".format(i, len(labels)))
            ground_truths = read_label_file(os.path.join(test_labels_dir, label_file_name))
            predictions = classifier.predict(image)
            num_predicted, num_missed = get_num_matching_predictions(ground_truths, predictions)
            true_positives += num_predicted
            false_negatives += num_missed
        if true_positives + false_negatives > 0:
            recall = float(true_positives) / float(true_positives + false_negatives)
            print("Recall: {}".format(recall))

    recall = float(true_positives) / float(true_positives + false_negatives)
    print("Final Recall: {}".format(recall))