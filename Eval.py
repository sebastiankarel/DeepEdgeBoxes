import tensorflow as tf
from Classification import Classification
import os
import cv2
import numpy as np
import xml.etree.ElementTree as et
from EdgeDetection import HED
import random
import time


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


def get_best_n(predictions, limits):
    result = []
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    for limit in limits:
        if len(predictions) > limit:
            result.append(predictions[:limit])
        else:
            result.append(predictions)
    return result


def get_num_matching_predictions(ground_truths, predictions, iou_threshold=0.5):
    best_predictions = []
    for ground_truth in ground_truths:
        best_prediction = (None, 0.0)
        for prediction in predictions:
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


def run_eval(sample, edge_type="single_canny"):
    log_file = open("eval_results_{}.txt".format(edge_type), 'x')
    log_file.write("Starting evaluation for {}...\n".format(edge_type))

    if edge_type == "multi_canny":
        weight_file = "bin_classifier_weights_multi.h5"
    elif edge_type == "rgb_canny":
        weight_file = "bin_classifier_weights_rgb.h5"
    elif edge_type == "hed":
        weight_file = "bin_classifier_weights_hed.h5"
    else:
        weight_file = "bin_classifier_weights.h5"

    use_hed = edge_type == "hed"
    use_multi = edge_type == "multi_canny"
    use_rgb = edge_type == "rgb_canny"

    if edge_type == "hed":
        hed = HED()
    else:
        hed = None

    classifier = Classification(224, 224, weight_file=weight_file, use_hed=use_hed, use_multichannel=use_multi, use_rgb=use_rgb, hed=hed)

    start = time.process_time()
    ious = [0.7, 0.6, 0.5, 0.4, 0.3]
    limits = [1000, 100, 10]
    true_positives = np.zeros((len(ious), len(limits)))
    false_negatives = np.zeros((len(ious), len(limits)))
    num_proposals = 0
    for i, label_file_name in enumerate(sample):
        image_file_name = label_file_name.split(".")[0] + ".jpg"
        image = cv2.imread(os.path.join(test_images_dir, image_file_name))
        if image is not None:
            ground_truths = read_label_file(os.path.join(test_labels_dir, label_file_name))
            predictions = classifier.predict(image)
            num_proposals += len(predictions)
            best_n_predictions = get_best_n(predictions, limits)
            for lim_idx, limited_predictions in enumerate(best_n_predictions):
                for iou_idx, iou in enumerate(ious):
                    num_predicted, num_missed = get_num_matching_predictions(ground_truths, limited_predictions, iou_threshold=iou)
                    true_positives[iou_idx][lim_idx] += num_predicted
                    false_negatives[iou_idx][lim_idx] += num_missed
    print("-------------------------{}--------------------------------".format(edge_type))
    log_file.write("-------------------------{}--------------------------------\n".format(edge_type))
    print("Average number of proposals: {}".format(float(num_proposals) / float(len(sample))))
    log_file.write("Average number of proposals: {}\n".format(float(num_proposals) / float(len(sample))))
    for iou_idx, iou in enumerate(ious):
        print("-----------------------IOU: {}--------------------------".format(iou))
        log_file.write("-----------------------IOU: {}--------------------------\n".format(iou))
        for lim_idx, lim in enumerate(limits):
            if true_positives[iou_idx][lim_idx] + false_negatives[iou_idx][lim_idx] > 0:
                recall = float(true_positives[iou_idx][lim_idx]) / (float(true_positives[iou_idx][lim_idx] + float(false_negatives[iou_idx][lim_idx])))
            else:
                recall = 0.0
            print("Recall for limit {}: {}".format(lim, recall))
            log_file.write("Recall for limit {}: {}\n".format(lim, recall))
    print("Elapsed time: {} minutes".format(round(float(time.process_time() - start) / 60.0)))
    log_file.write("Elapsed time: {} minutes\n".format(round(float(time.process_time() - start) / 60.0)))
    print("--------------------------------------------------------------------")
    log_file.write("--------------------------------------------------------------------\n")
    log_file.close()


if __name__ == "__main__":
    init_tf_gpu()

    sample_size = 100

    print("Read eval_configs.txt")
    file = open("eval_configs.txt", "r")
    lines = file.readlines()

    test_images_dir = ""
    test_labels_dir = ""

    for line in lines:
        split = line.split("=")
        if len(split) == 2:
            if split[0] == "test_images_dir":
                test_images_dir = split[1]
            elif split[0] == "test_labels_dir":
                test_labels_dir = split[1]
    test_images_dir = test_images_dir.strip()
    test_labels_dir = test_labels_dir.strip()
    labels = os.listdir(test_labels_dir)
    sample = random.sample(labels, sample_size)

    print("Starting Evaluation...")

    run_eval(sample, "single_canny")
    run_eval(sample, "multi_canny")
    run_eval(sample, "rgb_canny")
    run_eval(sample, "hed")

    print("Done.")
