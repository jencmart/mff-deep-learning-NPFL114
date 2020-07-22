

import tensorflow as tf

from svhn_competition import AnchorParams
from svhn_dataset import MyGenerator, SVHN
import numpy as np

import pandas as pd
import cv2
#
# # Automatic brightness and contrast optimization with optional histogram clipping
# def automatic_brightness_and_contrast(image, clip_hist_percent=25):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Calculate grayscale histogram
#     hist = cv2.calcHist([gray],[0],None,[256],[0,256])
#     hist_size = len(hist)
#
#     # Calculate cumulative distribution from the histogram
#     accumulator = []
#     accumulator.append(float(hist[0]))
#     for index in range(1, hist_size):
#         accumulator.append(accumulator[index -1] + float(hist[index]))
#
#     # Locate points to clip
#     maximum = accumulator[-1]
#     clip_hist_percent *= (maximum/100.0)
#     clip_hist_percent /= 2.0
#
#     # Locate left cut
#     minimum_gray = 0
#     while accumulator[minimum_gray] < clip_hist_percent:
#         minimum_gray += 1
#
#     # Locate right cut
#     maximum_gray = hist_size -1
#     while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
#         maximum_gray -= 1
#
#     # Calculate alpha and beta values
#     alpha = 255 / (maximum_gray - minimum_gray)
#     beta = -minimum_gray * alpha
#
#     '''
#     # Calculate new histogram with desired range and show histogram
#     new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
#     plt.plot(hist)
#     plt.plot(new_hist)
#     plt.xlim([0,256])
#     plt.show()
#     '''
#
#     auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#     return (auto_result, alpha, beta)
#
# image = cv2.imread('1.png')
# auto_result, alpha, beta = automatic_brightness_and_contrast(image)
# print('alpha', alpha)
# print('beta', beta)
# cv2.imshow('auto_result', auto_result)
# cv2.imwrite('auto_result.png', auto_result)
# cv2.imshow('image', image)
# cv2.waitKey()


def their_smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = tf.where(tf.keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = tf.keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
        normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
        return tf.keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true
        anchor_state = tf.math.reduce_sum(y_true, axis=2)

        # filter out "ignore" anchors
        indices   = tf.where(tf.keras.backend.not_equal(anchor_state, 0))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = tf.keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
        normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
        return tf.keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes). # batch, cnt , num_classes
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """

        ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=False)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        result = tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
        print("keras focal loss{}".format(result))

        # compute the focal loss
        alpha_factor = tf.keras.backend.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        modulating_factor = tf.where(tf.keras.backend.equal(y_true, 1), 1 - y_pred, y_pred) ** gamma
        cls_loss = tf.keras.backend.sum(alpha_factor * modulating_factor * ce)
        print("fizir focal loss{}".format(cls_loss))

        # compute the normalizer: the number of positive anchors
        anchor_state = y_true[:, :, 0]  # we have one hot encoded 0 as ignore ...  # -1 for ignore, 0 for background, 1 for object
        normalizer = tf.where(tf.keras.backend.equal(anchor_state, 1))
        normalizer = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
        normalizer = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

        print(normalizer)

        normalized = cls_loss / normalizer
        print("normalized fizir focal loss{}".format(normalized))


    return _focal


def generate_few_examples(anchor_params, cnt, side_size, tfrecords, specific=None):
    train_generator = MyGenerator(batch_size=1, dataset=tfrecords.test, shufle=False,
                                  anchor_params=anchor_params, test_only_few=True,
                                  side_size=side_size,
                                  resize_always=False,
                                  random_transform=False,
                                  to_grayscale=True,
                                  enhance=False)
    if specific:
        for i in specific:
            train_generator.__getitem__(i)
    else:
        for i in range(cnt):
            train_generator.__getitem__(i)
        #     print(i)
    #
    # res = train_generator.debug
    # print(res)
    #
    # res = pd.Series(res)
    # print(res)
    # print(res.describe())


if __name__ == "__main__":
    tfrecords = SVHN()


    #
    #              size of our boxes                  0.3      0.5   0.8
    #     image_min_side=800,                         339      565   905
    #     image_max_side=1333,                        565      942   1508
    #      their                 32    64    128      256   512

    side_size = 224  # /8
    levels = [5, 6, 7]
    optimal_ratios = np.array([0.5, 0.6, 0.8])  # height * ratio = width  --> [width/height]
    analysed_height_ratios = np.array([0.3, 0.5, 0.8])
    optimal_anchor_heights = analysed_height_ratios * side_size
    most_common_wh_ratio = 0.5
    # size = sqrt(H^2 / most_common_aspect ) ; 800 == [339 565 905]  ;  100 == [42  70 113]
    anchor_sizes_to_match_optimal_heights = np.sqrt(np.square(optimal_anchor_heights) * most_common_wh_ratio).astype(
        dtype=np.int)
   #  anchor_sizes_to_match_optimal_heights = np.array([30, 50, 65])
    optimal_strides = (anchor_sizes_to_match_optimal_heights / 4).astype(dtype=np.int)
    optimal_strides = optimal_strides
    anchor_sizes_to_match_optimal_heights = anchor_sizes_to_match_optimal_heights

    print("optimal sizes")
    print(anchor_sizes_to_match_optimal_heights)
    print("optimal str")
    print(optimal_strides)




    # anchor_params = AnchorParams(
    #     levels=[5, 6, 7],
    #     anchor_sizes=anchor_sizes_to_match_optimal_heights,
    #     anchor_strides=optimal_strides,  # /4
    #     ratios=optimal_ratios,
    #     scales=[2 ** 0]
    # )
    # generate_few_examples(anchor_params, 4, side_size)

    # default params
    # levels = [2, 4, 5]
    # anchor_strides = [7, 16, 32, ]
    # anchor_sizes = [15, 32, 50]
    # scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    # ratios = [0.5, 1, 2]

    anchor_params = AnchorParams(
        levels=[5, 6, 7],
        anchor_sizes=[35],
        anchor_strides=[10, 8, 14, ],
        scales=[2 ** 0, 2**1],
        ratios=[0.5, 1],
    )

    # 224 -- to bude nase finalni size
    # 50 --> 70 x 35  (optimalni  pro 0.3)
    # 80 --> 113 x 56 ( optimalni pro 0.5)
    # 100 --> 141 (0.5 ma mit 112)
    # 130 --> 183 x 91  (optimal pro 0.8)  [ ale presto nefunguje nejlepe]
    # 150 --> 212x106

    # [0.5, 0.6, 0.8]
    side_size = 64
    anchor_params = AnchorParams(
        levels=[3, 4, 5], #  4, 5 # 6, 7 ## pocitej s tim, ze na danem levelu mame male H,V takze celkovy pocet anchors je omezen !!!
        # level 3 --> 8,  4 --> 4,   5 --> 2
        #             64       16        4
        # pocet anchors == shape na levelu ^2 == 100 anchors
        anchor_sizes=[16, 32, 50], # 40, 60   #  80, 130
        anchor_strides=[8, 16, 32],  # 4x mensi strides to serou
        ratios=[0.3, 0.5, 0.7],  # 0.8
        scales=[2**1.2, 2**1,  2 ** (2.0 / 3.0)]
    )


    spec = [89, 216, 233, 298, 302, 650, 700, 728, 923 , 995 , 1123, 1181, 1255, 1461, 1665, 1797, 1938, 2330, 2428, 2495, 3214, 3268, 3489, 3596, 3825, 4204, 4387   ]
    spec = spec[-6:]
    generate_few_examples(anchor_params, 10, side_size, tfrecords, spec)
    exit(1)





    # anchor_params = AnchorParams(levels=[5,6,7],
    #                              ratios=optimal_ratios,
    #                              anchor_sizes=[14,30,14,14,14]
    #                              )




    #                               0   1   2
    classification = tf.constant([[.8, .3, .2],
                                  [.1, .1, .5],
                                  [.2, .1, .1],  # reset this row
                                  [.2, .7, .8]], dtype="float")
    boxes = tf.constant([[[1, 1, 2, 2, 1],
                           [1, 1, 0, 0, 0],
                           [3, 3, 7, 7, 1]],

                          [[2, 2, 2, 2, 1],
                           [3, 3, 4, 0, 1],
                           [0, 0, 0, 0, 0]],

                         [[3, 3, 2, 2, 1],
                          [3, 3, 4, 0, 1],
                          [0, 0, 0, 0, 0]],

                         [[4, 4, 2, 2, 1],
                          [3, 3, 4, 0, 1],
                          [0, 0, 0, 0, 0]],
                          ], dtype=tf.float32
                         )


    ind = tf.where(tf.greater(tf.argmax(classification, axis=1), 0))
    print(ind)
    classification = tf.gather_nd(classification, ind)
    boxes = tf.gather_nd(boxes, ind)

    print(boxes)

    exit(22)


    labels = tf.keras.backend.argmax(classification, axis=1)
    labels = tf.reshape(labels, (-1, 1))
    print(labels)
    exit(2)
    classification = tf.where()
    i = tf.cast(tf.keras.backend.flatten(tf.where(tf.keras.backend.equal(labels, 0))), dtype=tf.int32)




    exit(2)
    reset_constant = tf.repeat([[0.,0.,0.]], [4], axis=0)
    reset_constant = tf.where()
    print(reset_constant)
    exit(1)
    # tensor_with_reset_row = tf.mul(tensor_to_reset, reset_constant)






    y_true = tf.constant([[[0, 0, 2, 2, 1],
                           [0, 0, 0, 0, 0],
                           [3, 3, 7, 7, 1]],

                          [[0, 0, 2, 2, 1],
                           [3, 3, 4, 0, 1],
                           [0, 0, 0, 0, 0]]
                          ], dtype=tf.float32
                         )

    my_y_true = tf.constant([[[0, 0, 2, 2],
                              [0, 0, 0, 0],
                              [3, 3, 7, 7]],

                             [[0, 0, 2, 2],
                              [3, 3, 4, 0],
                              [0, 0, 0, 0]]
                             ], dtype=tf.float32
                            )

    y_pred = tf.constant([[[0, 0, 2, 2],
                           [0, 0, 0, 0],
                           [3, 3, 6, 6]],

                          [[0, 0, 2, 2],
                           [0, 0, 0, 0],
                           [3, 3, 6, 6]]
                          ], dtype=tf.float32
                         )

    res = their_smooth_l1()(y_true, y_pred)
    my_res = smooth_l1()(my_y_true, y_pred)

    # print(res)
    # print(my_res)

    y_t = tf.constant([[[0.97], [0.91], [0.03]]])
    y_p = tf.constant([[[1.0], [1.0], [0.0]]])
    focal()(y_t, y_p)