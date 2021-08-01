import cv2 as cv
import os
import numpy as np

# https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py
# https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/
# Caffe model https://github.com/s9xie/hed


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


class HED:

    def __init__(self):
        self.net = cv.dnn.readNetFromCaffe(os.path.join("hed", "deploy.prototxt"),
                                           os.path.join("hed", "hed_pretrained_bsds.caffemodel"))
        cv.dnn_registerLayer('Crop', CropLayer)

    def get_edge_image(self, input_image, width, height, normalized):
        image = cv.resize(input_image, (width, height))
        (mean_r, mean_g, mean_b, mean_x) = cv.mean(image)

        inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height),
                                   mean=(mean_r, mean_g, mean_b),
                                   swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()

        out = out[0, 0]
        out = cv.resize(out, (image.shape[1], image.shape[0]))

        if not normalized:
            out = out * 255

        return out


def auto_canny(src_image, multi_channel=False, rgb=False):
    image = src_image.copy()
    if rgb:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        upper_r, _ = cv.threshold(r, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        upper_g, _ = cv.threshold(g, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        upper_b, _ = cv.threshold(b, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        lower_r = upper_r * 0.5
        lower_g = upper_g * 0.5
        lower_b = upper_b * 0.5
        blurred_r = cv.GaussianBlur(r, (3, 3), 0)
        blurred_g = cv.GaussianBlur(g, (3, 3), 0)
        blurred_b = cv.GaussianBlur(b, (3, 3), 0)
        result_r = cv.Canny(blurred_r, lower_r, upper_r)
        result_g = cv.Canny(blurred_g, lower_g, upper_g)
        result_b = cv.Canny(blurred_b, lower_b, upper_b)
        result = np.dstack((result_r, result_g, result_b))
        return result
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        upper, _ = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        lower = upper * 0.5
        if multi_channel:
            low_sigma = cv.GaussianBlur(image, (3, 3), 0)
            high_sigma = cv.GaussianBlur(image, (5, 5), 0)
            no_sigma = cv.Canny(image, lower, upper)
            low_sigma = cv.Canny(low_sigma, lower, upper)
            high_sigma = cv.Canny(high_sigma, lower, upper)
            result = np.dstack((no_sigma, low_sigma, high_sigma))
            return result
        else:
            blurred = cv.GaussianBlur(image, (3, 3), 0)
            result = cv.Canny(blurred, lower, upper)
            return result
