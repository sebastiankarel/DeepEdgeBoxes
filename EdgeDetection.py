import cv2 as cv
import os

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
