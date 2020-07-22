#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import math

TOP, LEFT, BOTTOM, RIGHT = range(4)

class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):  # prob of fg [100 boxes 1 number]
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        dtype=np.float32
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)
        return result



class UpsampleLike(tf.keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = tf.keras.backend.shape(target)

        m = tf.image.ResizeMethod.NEAREST_NEIGHBOR

        return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), method=m)

    def compute_output_shape(self, input_shape):
        if tf.keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


def bbox_area(a):
    return max(0, a[BOTTOM] - a[TOP]) * max(0, a[RIGHT] - a[LEFT])


def bbox_iou(a, b):
    """ Compute IoU for two bboxes a, b.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).
    """
    intersection = [
        max(a[TOP], b[TOP]),
        max(a[LEFT], b[LEFT]),
        min(a[BOTTOM], b[BOTTOM]),
        min(a[RIGHT], b[RIGHT]),
    ]
    if intersection[RIGHT] <= intersection[LEFT] or intersection[BOTTOM] <= intersection[TOP]:
        return 0
    return bbox_area(intersection) / float(bbox_area(a) + bbox_area(b) - bbox_area(intersection))


def TLBR_to_center_hw(four_tuple):
    # # [0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0] # anchor, bbox, fast_rcnn
    x_r = (four_tuple[LEFT] + four_tuple[RIGHT]) / 2  # proc opacne ?????
    y_r = (four_tuple[TOP] + four_tuple[BOTTOM]) / 2

    w_r = four_tuple[RIGHT] - four_tuple[LEFT]  # TLBR
    h_r = four_tuple[BOTTOM] - four_tuple[TOP]
    assert w_r > 0
    assert h_r > 0

    return y_r, x_r, h_r, w_r


def bbox_to_fast_rcnn(anchor, bbox):
    """ Convert `bbox` to a Fast-R-CNN-like representation relative to `anchor`.

    The `anchor` and `bbox` are four-tuples (top, left, bottom, right);
    you can use SVNH.{TOP, LEFT, BOTTOM, RIGHT} as indices.

    The resulting representation is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - np.log(bbox_height / anchor_height)
    - np.log(bbox_width / anchor_width)

    BBOX relative to the anchor(RoI)
    """
    # TOP=0, LEFT=1, BOTTOM=2, RIGHT=3
    assert anchor[BOTTOM] > anchor[TOP]
    assert anchor[RIGHT] > anchor[LEFT]
    assert bbox[BOTTOM] > bbox[TOP]
    assert bbox[RIGHT] > bbox[LEFT]

    # [0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0] # anchor, bbox, fast_rcnn

    # calculate positions of ANCHOR and BBOX
    y_r, x_r, h_r, w_r = TLBR_to_center_hw(anchor)
    y, x, h, w = TLBR_to_center_hw(bbox)

    # BBOX relative to ANCHOR
    t_y = (y - y_r) / h_r
    t_x = (x - x_r) / w_r
    t_h = np.log(h / h_r)
    t_w = np.log(w / w_r)

    return t_y, t_x, t_h, t_w


def bbox_from_rcnn_keras(anchors, fast_rcnn):
    t_y = fast_rcnn[:, :, 0]
    t_x = fast_rcnn[:, :, 1]
    t_h = fast_rcnn[:, :, 2]
    t_w = fast_rcnn[:, :, 3]

    x_r = (anchors[:, :, LEFT] + anchors[:, :, RIGHT]) * 0.5  # proc opacne ?????
    y_r = (anchors[:, :, TOP] + anchors[:, :, BOTTOM]) * 0.5

    w_r = anchors[:, :, RIGHT] - anchors[:, :, LEFT]  # TLBR
    h_r = anchors[:, :, BOTTOM] - anchors[:, :, TOP]

    y = t_y * h_r + y_r
    x = t_x * w_r + x_r
    h = tf.keras.backend.exp(t_h) * h_r
    w = tf.keras.backend.exp(t_w) * w_r

    top = tf.keras.backend.round(y - h * 0.5)
    left = tf.keras.backend.round(x - w * 0.5)
    bottom = tf.keras.backend.round(y + h * 0.5)
    right = tf.keras.backend.round(x + w * 0.5)

    boxes = tf.keras.backend.stack([top, left, bottom, right], axis=2)
    return boxes


# tahle funkce == regression boxes (aka bbox_transform_inv)
def bbox_from_fast_rcnn(anchor, fast_rcnn):
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`."""
    s = "chyba: bottom: {}  , top {}".format(anchor[BOTTOM], anchor[TOP])
    assert anchor[BOTTOM] >= anchor[TOP], s
    assert anchor[RIGHT] >= anchor[LEFT]

    # calculate positions of ANCHOR and BBOX
    y_r, x_r, h_r, w_r = TLBR_to_center_hw(anchor)
    t_y, t_x, t_h, t_w = fast_rcnn

    # inverse the relative representation
    y = t_y * h_r + y_r
    x = t_x * w_r + x_r
    h = np.exp(t_h) * h_r
    w = np.exp(t_w) * w_r

    # calculate corners from center, w, h
    top = np.round(y - h / 2)
    bottom = np.round(y + h / 2)
    left = np.round(x - w / 2)
    right = np.round(x + w / 2)

    return top, left, bottom, right


def sort_the_shit(list_of_items, one_item):
    # calculate IoU of of this gold_bbox with a
    # ll anchors
    ious = np.zeros(len(list_of_items), np.float32)
    for i, anchor in enumerate(list_of_items):
        #  Each bbox is parametrized as a four-tuple (top, left, bottom, right).
        ious[i] = bbox_iou(anchor, one_item)

    # sort them by the IoU [-1 because we want descending order and args-ort does not support that]
    sorted_idx = np.argsort(-1 * ious, kind="mergesort")
    return ious, sorted_idx


# my trenujeme na anchors a ne na gold classes, right ????
# a ty anchors si musime vygenerovat pomoci teto funkce ...

# oni vraci regression-batch, label-batch
def bboxes_training(anchors, gold_classes, gold_bboxes, iou_threshold):
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)

    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right) of the gold objects

    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes`
        for every anchor
            0 (no gold object is assigned)
            or `1 + gold_class` if a gold object
      with `gold_class` as assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN
      zeros if no gold object was assigned to the anchor

    Algorithm:
    - For each gold object,
      find the first unused anchor with largest IoU
       if the IoU is > 0,  assign the object to the anchor.

    - Second, anchors unassigned so far are sequentially processed.
      For each anchor, find the first gold object with the largest IoU,
      if IoU >= threshold, assign the object to the anchor.
    """

    anchor_classes = np.zeros(len(anchors), np.int32)
    anchor_bboxes = np.zeros([len(anchors), 4], np.float32)
    # print(anchors.shape)  # 70 boxes

    #  Sequentially for each gold object, find the first unused anchor
    # with the largest IoU and if the IoU is > 0, assign the object to the anchor.
    for gold_class, gold_bbox in zip(gold_classes, gold_bboxes):

        ious, sorted_idx_by_iou = sort_the_shit(anchors, gold_bbox)

        for i in sorted_idx_by_iou:
            if anchor_classes[i] == 0 and np.greater(ious[i], 0):  # and only if free and > 0
                # print("1 assign")
                anchor_classes[i] = gold_class + 1  # because we want to be shifted ...
                anchor_bboxes[i] = bbox_to_fast_rcnn(anchors[i], gold_bbox)
                break

    # Sequentially for each unassigned anchor
    # find the first gold object with the largest IoU.
    # If the IoU >= threshold, assign the object to the anchor.
    for j, anchor in enumerate(anchors):
        if anchor_classes[j] == 0:  # not yet assigned
            # print("Not yet assigned")
            ious, sorted_idx_by_iou = sort_the_shit(gold_bboxes, anchor)

            for i in sorted_idx_by_iou:  # 3, 7, 2, 15, 0 ... they are true indexes of anchors ...
                # print("trying to assing {}".format(ious[i]))
                if np.greater_equal(ious[i], iou_threshold):
                    anchor_classes[j] = gold_classes[i] + 1  # because we want to be shifted
                    anchor_bboxes[j] = bbox_to_fast_rcnn(anchor, gold_bboxes[i])
                    # print("assigning box")
                    break

    return anchor_classes, anchor_bboxes



def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride # svisle
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride  # vodorovne
    # print(shape[1])
    # print(shape[0])
    # print(shift_x)
    # print(shift_y)
    # exit(99)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()
    # print("total shape {}".format(shifts.shape))

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(desired_sizes, ratios, scales):
    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = desired_sizes * np.tile(scales, (2, len(ratios))).T  # pro kazdej ratio a kazdej scale

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    #
    # # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # print(anchors[0])
    # # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # start:stop:step
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T  # 0, 1  x1 = xctr - w * 0.5  x2 = w - w * 0.5
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T  # 2, 3
    # print(anchors[0])
    return anchors


def anchors_for_shape(max_shape,
                      ratios,
                      scales,
                      sizes,
                      strides,
                      pyramid_levels):
    all_anchors = np.zeros((0, 4))
    # print(max_shape)
    max_shape = np.array(max_shape[:2])  # [(w,h), (w,h)  ]

    shapes = []

    for x in pyramid_levels:
        shapes.append((max_shape + 2 ** x - 1) // (2 ** x))  # integer division
    # print(shapes)
    # exit(22)
    # print(shapes)
    # exit(23)
    # shapes = [[7, 7]]
    # for each level of pyramid ...
    for idx_level in range(len(pyramid_levels)):
        # sizes of anchors in the final image ? yup definitley ...

        anchors = generate_anchors(sizes[idx_level], ratios=ratios, scales=scales)  # sizes 0

        # mame batch images, kazdy je jinak veliky
        # image_shape = max_w, max_h, max_ch .. nezavisle

        # to jsou myslim true shapes ktere lezou z daneho levelu ... (klesaji ... )
        # guess shapes of image at the certain level

        # rint(shapes)  # poji se k levelu   28  14    7    4    2 [klesaji ... ]
        shifted_anchors = shift(shapes[idx_level], strides[idx_level], anchors)  ## todo -- depends on shapes, strides
        # print(shifted_anchors[0])
        # print(shifted_anchors[100])
        # exit(1)
        # print(shifted_anchors.shape)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
    return all_anchors




def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = False,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 3,
    nms_threshold         = 0.5
):

    def _filter_detections(scores, labels):
        # threshold based on score
        indices = tf.where(tf.keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = tf.gather_nd(boxes, indices)
            filtered_scores = tf.keras.backend.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = tf.keras.backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = tf.keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((tf.keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = tf.keras.backend.concatenate(all_indices, axis=0)
    else:
        scores  = tf.keras.backend.max(classification, axis    = 1)
        labels  = tf.keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = tf.keras.backend.gather(indices[:, 0], top_indices)
    boxes               = tf.keras.backend.gather(boxes, indices)
    labels              = tf.keras.backend.gather(labels, top_indices)
    labels = labels - 1 # todo ---- ted jdeme 0 az 9 a ne 1 az 10 ...
    other_              = [tf.keras.backend.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = tf.keras.backend.maximum(0, max_detections - tf.keras.backend.shape(scores)[0])
    boxes    = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = tf.keras.backend.cast(labels, 'int32')
    other_   = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(tf.keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels]


#
# def filter_detections(
#         boxes,
#         classification,
#         other=[],
#         class_specific_filter=True,
#         nms=True,
#         score_threshold=0.05,
#         max_detections=300,
#         nms_threshold=0.5
# ):
#     """ Filter detections using the boxes and classification values.
#
#     Args
#         boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
#         classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
#         other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
#         class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
#         nms                   : Flag to enable/disable non maximum suppression.
#         score_threshold       : Threshold used to prefilter the boxes with.
#         max_detections        : Maximum number of detections to keep.
#         nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
#
#     Returns
#         A list of [boxes, scores, labels, other[0], other[1], ...].
#         boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
#         scores is shaped (max_detections,) and contains the scores of the predicted class.
#         labels is shaped (max_detections,) and contains the predicted label.
#         other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
#         In case there are less than max_detections detections, the tensors are padded with -1's.
#     """
#
#     def _filter_detections(scores, labels):
#         # threshold based on score
#         indices = tf.where(tf.keras.backend.greater(scores, score_threshold))
#
#         if nms:
#             filtered_boxes = tf.gather_nd(boxes, indices)
#             filtered_scores = tf.keras.backend.gather(scores, indices)[:, 0]
#
#             # perform NMS
#             nms_indices = tf.image.non_max_suppression(filtered_boxes,
#                                                        filtered_scores,
#                                                        max_output_size=max_detections,
#                                                        iou_threshold=nms_threshold)
#
#             # filter indices based on NMS
#             indices = tf.keras.backend.gather(indices, nms_indices)
#
#         # add indices to list of all indices
#         labels = tf.gather_nd(labels, indices)
#         indices = tf.keras.backend.stack([indices[:, 0], labels], axis=1)
#
#         return indices
#
#     # if class_specific_filter:
#     all_indices = []
#     # perform per class filtering
#     for c in range(int(classification.shape[1])):  # 0 batch, 1 probs ...
#         scores = classification[:, c]  # just probability .... for all of this class
#         if c == 0:  # background have probability zero ....
#             scores = c * scores
#         labels = c * tf.ones((tf.keras.backend.shape(scores)[0],), dtype='int64')  # labels ... 0 ... 10
#         all_indices.append(_filter_detections(scores, labels))  # todo -- call here # scores and labels of one class --- over all
#     # concatenate indices to single tensor
#     indices = tf.keras.backend.concatenate(all_indices, axis=0)
#
#     # else:
#     # todo -- jen pokus s timto
#     scores = tf.keras.backend.max(classification, axis=1)
#     labels = tf.keras.backend.argmax(classification, axis=1)
#     indices = _filter_detections(scores, labels)
#
#     # select top k
#     scores = tf.gather_nd(classification, indices)
#     labels = indices[:, 1]
#     scores, top_indices = tf.nn.top_k(scores,
#                                       k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))
#
#     # --------- Final filtering based on score --------------------------
#     # filter input using the final set of indices
#     indices = tf.keras.backend.gather(indices[:, 0], top_indices)
#     boxes = tf.keras.backend.gather(boxes, indices)
#     labels = tf.keras.backend.gather(labels, top_indices)
#
#     labels = labels - 1
#
#     # zero pad the outputs
#     pad_size = tf.keras.backend.maximum(0, max_detections - tf.keras.backend.shape(scores)[0])
#     boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
#     scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
#     labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
#     labels = tf.keras.backend.cast(labels, 'int32')

#     # set shapes, since we know what they are
#     boxes.set_shape([max_detections, 4])
#     scores.set_shape([max_detections])
#     labels.set_shape([max_detections])
#
#     return [boxes, scores, labels]


class FilterDetections(tf.keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.05,
            max_detections=300,
            parallel_iterations=32,
            preliminary_remove_bg_class='no',
            **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.preliminary_remove_bg_class = preliminary_remove_bg_class

        assert self.preliminary_remove_bg_class in ['zero-score', 'remove-rows', 'no'], "invalid option choose 'zero-score' or 'remove-rows' or 'no' for no prelimiary removing"
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        other = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes = args[0]
            classification = args[1]
            other = args[2]

            if self.preliminary_remove_bg_class == 'zero-score':
                # Option 1 --- zero-out first column
                col_to_zero = [0]  # <-- column numbers you want to be zeroed out
                tnsr_shape = tf.shape(classification)
                mask = [tf.one_hot(col_num * tf.ones((tnsr_shape[0],), dtype=tf.int32), tnsr_shape[-1])
                        for col_num in col_to_zero]
                mask = tf.reduce_sum(mask, axis=0)
                mask = tf.cast(tf.logical_not(tf.cast(mask, tf.bool)), tf.float32)
                classification = classification * mask

            # Another option -- remove rows where highest probability is BG
            if self.preliminary_remove_bg_class == 'remove-rows':
                i = tf.where(tf.greater(tf.argmax(classification, axis=1), 0))
                boxes = tf.gather_nd(boxes, i)  # (None, 4)
                classification = tf.gather_nd(classification, i)  # (None, 11)
                # tf.print(classification, [classification])

            return filter_detections(
                boxes,
                classification,
                other,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[tf.keras.backend.floatx(), tf.keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ]
        #        + [
        #     tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        # ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config


class RegressBoxes(tf.keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_from_rcnn_keras(anchors, regression)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        })

        return config


class ClipBoxes(tf.keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = tf.keras.backend.cast(tf.keras.backend.shape(image), tf.keras.backend.floatx())

        _, height, width, _ = tf.unstack(shape, axis=0)
        x1, y1, x2, y2 = tf.unstack(boxes, axis=-1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)
        x2 = tf.clip_by_value(x2, 0, width - 1)
        y2 = tf.clip_by_value(y2, 0, height - 1)

        return tf.keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def sshift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (tf.keras.backend.arange(0, shape[1], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5,
                                                                                                                 dtype=tf.keras.backend.floatx())) * stride
    shift_y = (tf.keras.backend.arange(0, shape[0], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5,
                                                                                                                 dtype=tf.keras.backend.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.keras.backend.reshape(shift_x, [-1])
    shift_y = tf.keras.backend.reshape(shift_y, [-1])

    shifts = tf.keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = tf.keras.backend.transpose(shifts)
    number_of_anchors = tf.keras.backend.shape(anchors)[0]

    k = tf.keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = tf.keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + tf.keras.backend.cast(
        tf.keras.backend.reshape(shifts, [k, 1, 4]), tf.keras.backend.floatx())
    shifted_anchors = tf.keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


class Anchors(tf.keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
            scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if isinstance(scales, list):
            self.scales = np.array(scales)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors = tf.keras.backend.variable(generate_anchors(
            desired_sizes=self.size,
            ratios=self.ratios,
            scales=self.scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = tf.keras.backend.shape(features)

        # generate proposals from bbox deltas and shifted anchors
        anchors = sshift(features_shape[1:3], self.stride, self.anchors)
        anchors = tf.keras.backend.tile(tf.keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if tf.keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
        })

        return config


def build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        Anchors(
            size=anchor_parameters.anchor_sizes[i],
            stride=anchor_parameters.anchor_strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return tf.keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def compute_targets(gold_classes, gold_boxesTLBR, max_shape, ratios, scales, sizes, strides, levels, iou_threshold):
    # instead of bboxes we want fucking anchor targets

    anchorsTLBR = anchors_for_shape(max_shape=max_shape,
                                    ratios=ratios,
                                    scales=scales,
                                    sizes=sizes,
                                    strides=strides,
                                    pyramid_levels=levels)

    computed_classes, computed_bboxes = bboxes_training(anchorsTLBR, gold_classes, gold_boxesTLBR, iou_threshold)

    # computed_classes[0] = 10

    shape = (computed_classes.size, 10 + 1)  # computed_classes.max()
    one_hot = np.zeros(shape, dtype=np.float32)
    rows = np.arange(computed_classes.size)
    one_hot[rows, computed_classes] = 1

    # i = 0
    # for one, val in zip(computed_bboxes, computed_classes):
    #     if val != 0:
    #         i+=1# print("-------")
    #         # print(val)
    # print(one)
    # exit(32)
    # print(" nonzero {}".format(i))
    # print(computed_bboxes.shape)
    return one_hot, computed_bboxes, anchorsTLBR  # computed_classes

    # np_img = data[0]['image'].numpy()
    # np_img = np.reshape(np_img, (76, 76, 3))

    #                p3  p4  p5  |  p6   p7
    #  -- u nas ---  56  28  14  |  7    4
    #  -- u nich --- 28  14   7  |  4    2

    # to se poji k levelu [pro gen. 9 basic anchors] (how big final anchors..)
    # to se poji k levelu [pro stride...]

    # levels = [3]
    # sizes = [15]
    # strides = [8]
    # anchors are same for the whole batch.... ( based on max shape )

    # printable_bboxes = list_TLBR_to_printable_x1y1_hw(gold_boxesTLBR)
    # anchor bboxes are now in rcnn representation, relative to the gold Boxes

    # filtered_anchorTLBR = []
    # for tlbr, box in zip(anchorsTLBR, computed_bboxes):
    #     if box[0] != 0. and box[1] != 0 and box[2] != 0 and box[3] != 0:
    #         print("ok")
    #
    #         filtered_anchorTLBR.append(tlbr)  # bbox_from_fast_rcnn(tlbr, box)

    # print(filtered_anchorTLBR)
    # print("bboxes TLBR")
    # print(gold_boxesTLBR)
    # exit(23)
    # printable_anchors = list_TLBR_to_printable_x1y1_hw(filtered_anchorTLBR)
    # print_image_with_rectangles(np_img, printable_bboxes, printable_anchors)


# import unittest

#
# class Tests(unittest.TestCase):
#     def test_bbox_to_from_fast_rcnn(self):
#         for anchor, bbox, fast_rcnn in [  # [0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]
#             [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
#             [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
#             [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
#             [[0, 0, 10, 10], [0, 0, 20, 20], [.5, .5, np.log(2), np.log(2)]],
#         ]:
#             np.testing.assert_almost_equal(bbox_to_fast_rcnn(anchor, bbox), fast_rcnn, decimal=3)
#             np.testing.assert_almost_equal(bbox_from_fast_rcnn(anchor, fast_rcnn), bbox, decimal=3)
#
#     def test_bboxes_training(self):
#         anchors = [[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]]
#         for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
#             [[1],
#              [[14, 14, 16, 16]],
#              [0, 0, 0, 2],
#              [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(1 / 5), np.log(1 / 5)]], 0.5],
#
#             [[2], [[0, 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
#
#             [[2], [[0, 0, 20, 20]], [3, 3, 3, 3], [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]],
#              0.24],
#
#         ]:
#             computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
#             np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
#             np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)
#
#
# if __name__ == '__main__':
#     unittest.main()

    # x: 0,  0,  0,  0,  5,  0,  0,  0,  0,  4,  0,  0,  7,  0,  3,  0,  0, 7,  0,  0,  0,  8, 10,  0, 10
    # y: 0,  0,  0,  0,  5,  0,  0,  0,  0,  4,  0,  0,  7,  0,  3,  0,  0, 7, 10,  0,  0,  8, 10,  0,  0
    # print("")

    #  x: array([ 0,  0,  0,  0,  5,  0,  0,  0,  0,  4,  0,  0,  7,  0,  3,  0,  0,  7,  0,  0,  0,  8, 10,  0, 10], dtype=int32)
    #  y: array([ 0,  0,  0,  0,  5,  0,  0,  0,  0,  4,  0,  0,  7,  0,  3,  0,  0, 7,  10,  0,  0,  8, 10,  0,  0], dtype=int32)
