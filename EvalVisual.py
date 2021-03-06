import tensorflow as tf
from Classification import Classification
import sys
import os
import cv2
import numpy as np
import xml.etree.ElementTree as et
from EdgeDetection import HED


def init_tf_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)


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


def compute_iou(ground_truth, prediction):
    x_overlap = max(0, min(ground_truth[2], prediction[2]) - max(ground_truth[0], prediction[0]))
    y_overlap = max(0, min(ground_truth[3], prediction[3]) - max(ground_truth[1], prediction[1]))
    intersection = float(x_overlap * y_overlap)
    gt_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
    pred_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    union = float(gt_area + pred_area) - intersection
    return intersection / union


def get_best_predictions(ground_truths, predictions, iou_threshold=0.5):
    result = []
    for ground_truth in ground_truths:
        best_prediction = (None, 0.0)
        for prediction in predictions:
            iou = compute_iou(ground_truth, prediction)
            if iou >= iou_threshold:
                if iou > best_prediction[1]:
                    best_prediction = (prediction, iou)
        result.append(best_prediction[0])
    return result


if __name__ == "__main__":
    init_tf_gpu()

    edge_type = "single_canny"
    for arg in sys.argv:
        split = arg.split("=")
        if len(split) == 2:
            if split[0] == "edge_type":
                if split[1] == "single_canny" or split[1] == "multi_canny" or split[1] == "hed" or split[1] == "rgb_canny":
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
    original_images_dir = ""

    if edge_type == "multi_canny":
        weight_file = "weights_multi.h5"
    elif edge_type == "rgb_canny":
        weight_file = "weights_rgb.h5"
    elif edge_type == "hed":
        weight_file = "weights_hed.h5"
    else:
        weight_file = "weights.h5"

    use_hed = edge_type == "hed"
    use_multi = edge_type == "multi_canny"
    use_rgb = edge_type == "rgb_canny"

    for line in lines:
        split = line.split("=")
        if len(split) == 2:
            if split[0] == "test_images_dir":
                test_images_dir = split[1]
            elif split[0] == "test_labels_dir":
                test_labels_dir = split[1]

    hed = HED()
    classifier = Classification(224, 224, weight_file=weight_file, use_hed=use_hed, use_multichannel=use_multi, use_rgb=use_rgb, hed=hed)
    classifier.set_model_for_prediction()

    print("Starting evaluation")
    test_images_dir = test_images_dir.strip()
    test_labels_dir = test_labels_dir.strip()
    labels = os.listdir(test_labels_dir)

    for i, label_file_name in enumerate(labels):
        image_file_name = label_file_name.split(".")[0] + ".jpg"
        image = cv2.imread(os.path.join(test_images_dir, image_file_name))
        if image is not None:
            print("Evaluating image {} of {}".format(i, len(labels)))
            ground_truths = read_label_file(os.path.join(test_labels_dir, label_file_name))
            predictions = classifier.predict(image)
            best_predictions = get_best_predictions(ground_truths, predictions)
            for prediction in best_predictions:
                if prediction is not None:
                    cv2.rectangle(image, (prediction[0], prediction[1]), (prediction[2], prediction[3]), (0, 255, 0), 1)
            for ground_truth in ground_truths:
                cv2.rectangle(image, (ground_truth[0], ground_truth[1]), (ground_truth[2], ground_truth[3]), (255, 0, 0), 1)
            cv2.imshow("result", image)
            cv2.waitKey()
