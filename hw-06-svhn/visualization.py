import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import tensorflow as tf

# image shapes == guess shapes( image_shape, pyramid_level)
# == list of shapes for each level
# == jaka bude sirka a vyska obrazku na danem levelu pyramidy ...
# from svhn_dataset import SVHN

import bboxes_utils as utils #  import bboxes_training
# from bboxes_utils import TLBR_to_center_hw
# from bboxes_utils import bbox_from_fast_rcnn
# from svhn_dataset import SVHN

TOP, LEFT, BOTTOM, RIGHT = range(4)


def print_boxes(ax, colors, boxes, labels=None, scores=None):
    for i in range(boxes.shape[0]):
        color = colors[i % len(colors)]
        # print(boxes[i])
        h = boxes[i, 2]
        w = boxes[i, 3]
        y = boxes[i, 0]
        x = boxes[i, 1]

        rect = Rectangle((x, y), width=w, height=h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        if labels is not None and scores is not None:
            s = ""
            s += str(labels[i])
            s += ";p:"
            score = str(scores[i])
            if len(score) > 4:
                score = score[:4]
            s += score

            ax.text(x+2, y+3, s=s,  bbox=dict(facecolor=color, alpha=0.8))


def print_image_with_rectangles(img, gold_bboxes, predicted_labels=None, predicted_scores=None,
                                predicted_boxes=None, only_predictions=False, print_box_sizes=False):

    fig, ax = plt.subplots(1)
    ax.imshow(img) # cmap='gray'
    # only_predictions = False True  # todo
    if not only_predictions:
        gold_bboxes = numpy_TLBR_to_printable_x1y1_hw(gold_bboxes, prnt=print_box_sizes)

        # print gold boxes
        colors = ['c', 'm', 'y']
        print_boxes(ax, colors, gold_bboxes)

    if predicted_boxes is not None:
        predicted_boxes = numpy_TLBR_to_printable_x1y1_hw(predicted_boxes, prnt=print_box_sizes)

        # print predicted boxes
        colors = ['b', 'g', 'r',  'w']
        print_boxes(ax, colors, predicted_boxes, predicted_labels, predicted_scores)

    plt.show()

def numpy_TLBR_to_printable_x1y1_hw(data, prnt=True):
    if prnt:
        w_r = data[:, RIGHT] - data[:, LEFT]  # TLBR
        h_r = data[:, BOTTOM] - data[:, TOP]
        # print("width")
        # print(w_r)
        # print("height")
        # print(h_r)
        # print("square")
        # print(w_r*h_r)
        # print("ratio")
        # print(w_r/h_r)

    # T L B R
    corrected = np.zeros((data.shape[0], 4))
    corrected[:, 0] = (data[:, TOP] + data[:, BOTTOM]) * 0.5  # y
    corrected[:, 1] = (data[:, LEFT] + data[:, RIGHT]) * 0.5  # x
    corrected[:, 2] = (data[:, BOTTOM] - data[:, TOP])  # h
    corrected[:, 3] = (data[:, RIGHT] - data[:, LEFT])  # w

    corrected[:, 0] = corrected[:, 0] - corrected[:, 2] * 0.5  # y_r = y_r - h_r/2
    corrected[:, 1] = corrected[:, 1] - corrected[:, 3] * 0.5  # x_r = x_r - h_r/2
    return corrected

def list_TLBR_to_printable_x1y1_hw(rectangles):
    corrected = np.zeros((0, 4))

    for rectangle in rectangles:
        y_r, x_r, h_r, w_r = utils.TLBR_to_center_hw(rectangle)
        y_r -= h_r / 2
        x_r -= w_r / 2
        corrected = np.append(corrected, [[y_r, x_r, h_r, w_r]], axis=0)

    return corrected


# def add_bbox_output(model):


if __name__ == "__main__":
    pass
    # train_len = 5115000
    # dev_len   = 85078
    # test_len  = 1055520

    # tfrecords = utils.SVHN()
    # train_generator = MyGenerator(5115000, batch_size=1,dataset= tfrecords.train, shufle=True)
    # dev_generator = MyGenerator(85078, batch_size=1, dataset=tfrecords.dev, shufle=True)



