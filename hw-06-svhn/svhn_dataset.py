import math
import os
import random
import sys
import urllib.request
from time import sleep

import cv2
import tensorflow as tf
import numpy as np

import bboxes_utils as utils
from visualization import print_image_with_rectangles

import matplotlib.pyplot as plt

DEFAULT_PRNG = np.random


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )
    return output

def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result





def colvec(*args):
    """ Create a numpy array representing a column vector. """
    return np.array([args]).T


def transform_aabb(transform, aabb):
    """ Apply a transformation to an axis aligned bounding box.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying the given transformation.

    Args
        transform: The transformation to apply.
        x1:        The minimum x value of the AABB.
        y1:        The minimum y value of the AABB.
        x2:        The maximum x value of the AABB.
        y2:        The maximum y value of the AABB.
    Returns
        The new AABB as tuple (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def _random_vector(min, max, prng=DEFAULT_PRNG):
    """ Construct a random vector between min and max.
    Args
        min: the minimum value for each component
        max: the maximum value for each component
    """
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return prng.uniform(min, max)


def rotation(angle):
    """ Construct a homogeneous 2D rotation matrix.
    Args
        angle: the angle in radians
    Returns
        the rotation matrix as 3 by 3 numpy array
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])



def noisy(noise_typ,image, ammount=0.3):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape((row,col,ch))
        noisy = image + gauss*ammount
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape((row,col,ch)) * ammount
        noisy = image + image * gauss
        return noisy

def random_rotation(min, max, prng=DEFAULT_PRNG):
    """ Construct a random rotation between -max and max.
    Args
        min:  a scalar for the minimum absolute angle in radians
        max:  a scalar for the maximum absolute angle in radians
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 rotation matrix
    """
    return rotation(prng.uniform(min, max))


def translation(translation):
    """ Construct a homogeneous 2D translation matrix.
    # Arguments
        translation: the translation 2D vector
    # Returns
        the translation matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def random_translation(min, max, prng=DEFAULT_PRNG):
    """ Construct a random 2D translation between min and max.
    Args
        min:  a 2D vector with the minimum translation for each dimension
        max:  a 2D vector with the maximum translation for each dimension
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 translation matrix
    """
    return translation(_random_vector(min, max, prng))


def shear(angle):
    """ Construct a homogeneous 2D shear matrix.
    Args
        angle: the shear angle in radians
    Returns
        the shear matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, -np.sin(angle), 0],
        [0,  np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_shear(min, max, prng=DEFAULT_PRNG):
    """ Construct a random 2D shear matrix with shear angle between -max and max.
    Args
        min:  the minimum shear angle in radians.
        max:  the maximum shear angle in radians.
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 shear matrix
    """
    return shear(prng.uniform(min, max))


def scaling(factor):
    """ Construct a homogeneous 2D scaling matrix.
    Args
        factor: a 2D vector for X and Y scaling
    Returns
        the zoom matrix as 3 by 3 numpy array
    """
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def random_scaling(min, max, prng=DEFAULT_PRNG):
    """ Construct a random 2D scale matrix between -max and max.
    Args
        min:  a 2D vector containing the minimum scaling factor for X and Y.
        min:  a 2D vector containing The maximum scaling factor for X and Y.
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 scaling matrix
    """
    return scaling(_random_vector(min, max, prng))


def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
    """ Construct a transformation randomly containing X/Y flips (or not).
    Args
        flip_x_chance: The chance that the result will contain a flip along the X axis.
        flip_y_chance: The chance that the result will contain a flip along the Y axis.
        prng:          The pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 transformation matrix
    """
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    # 1 - 2 * bool gives 1 for False and -1 for True.
    return scaling((1 - 2 * flip_x, 1 - 2 * flip_y))


def change_transform_origin(transform, center):
    """ Create a new transform representing the same transformation,
        only with the origin of the linear part changed.
    Args
        transform: the transformation matrix
        center: the new origin of the transformation
    Returns
        translate(center) * transform * translate(-center)
    """
    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])


def enhance_transform():
    pass

def random_transform(
    min_rotation=0,
    max_rotation=0,
    min_translation=(0, 0),
    max_translation=(0, 0),
    min_shear=0,
    max_shear=0,
    min_scaling=(1, 1),
    max_scaling=(1, 1),
    flip_x_chance=0,
    flip_y_chance=0,
    prng=DEFAULT_PRNG
):
    """ Create a random transformation.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        min_rotation:    The minimum rotation in radians for the transform as scalar.
        max_rotation:    The maximum rotation in radians for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear:       The minimum shear angle for the transform in radians.
        max_shear:       The maximum shear angle for the transform in radians.
        min_scaling:     The minimum scaling for the transform as 2D column vector.
        max_scaling:     The maximum scaling for the transform as 2D column vector.
        flip_x_chance:   The chance (0 to 1) that a transform will contain a flip along X direction.
        flip_y_chance:   The chance (0 to 1) that a transform will contain a flip along Y direction.
        prng:            The pseudo-random number generator to use.
    """
    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, prng),
        random_translation(min_translation, max_translation, prng),
        random_shear(min_shear, max_shear, prng),
        random_scaling(min_scaling, max_scaling, prng),
        random_flip(flip_x_chance, flip_y_chance, prng)
    ])


def enhance_generator(**kwargs):
    while True:
        yield enhance_transform( **kwargs)

def random_transform_generator(prng=None, **kwargs):
    """ Create a random transform generator.

    Uses a dedicated, newly created, properly seeded PRNG by default instead of the global DEFAULT_PRNG.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        min_rotation:    The minimum rotation in radians for the transform as scalar.
        max_rotation:    The maximum rotation in radians for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear:       The minimum shear angle for the transform in radians.
        max_shear:       The maximum shear angle for the transform in radians.
        min_scaling:     The minimum scaling for the transform as 2D column vector.
        max_scaling:     The maximum scaling for the transform as 2D column vector.
        flip_x_chance:   The chance (0 to 1) that a transform will contain a flip along X direction.
        flip_y_chance:   The chance (0 to 1) that a transform will contain a flip along Y direction.
        prng:            The pseudo-random number generator to use.
    """

    if prng is None:
        # RandomState automatically seeds using the best available method.
        prng = np.random.RandomState()

    while True:
        yield random_transform(prng=prng, **kwargs)

def resize_image(immg, min_size):
    scale = immg.shape[0] / min_size
    dim = (min_size, min_size)

    img = cv2.resize(immg, dim)  # , fx=scale, fy=scale)
    return img, scale




def _clip(image, normalized=True):
    """
    Clip and convert an image to np.uint8.

    Args
        image: Image to clip.
    """
    if normalized:
        return np.clip(image, 0., 1.).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)



def adjust_contrast(image, factor):


    """ Adjust contrast of an image.

    Args
        image: Image to adjust.
        factor: A factor for adjusting contrast.
    """
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta, normalized=True):
    """ Adjust brightness of an image

    Args
        image: Image to adjust.
        delta: Brightness offset between -1 and 1 added to the pixel values.
    """
    if normalized:
        return _clip(image + delta)

    return _clip(image + delta * 255)


def adjust_hue(image, delta):
    """ Adjust hue of an image.

    Args
        image: Image to adjust.
        delta: An interval between -1 and 1 for the amount added to the hue channel.
               The values are rotated if they exceed 180.
    """
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    """ Adjust saturation of an image.

    Args
        image: Image to adjust.
        factor: An interval for the factor multiplying the saturation values of each pixel.
    """
    image[..., 1] = np.clip(image[..., 1] * factor, 0 , 255)
    return image


class VisualEffect:
    """ Struct holding parameters and applying image color transformation.

    Args
        contrast_factor:   A factor for adjusting contrast. Should be between 0 and 3.
        brightness_delta:  Brightness offset between -1 and 1 added to the pixel values.
        hue_delta:         Hue offset between -1 and 1 added to the hue channel.
        saturation_factor: A factor multiplying the saturation values of each pixel.
    """

    def __init__(
        self,
        contrast_factor,
        brightness_delta,
        hue_delta,
        saturation_factor,
        gauss_noise
    ):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor
        self.gauss_noise = gauss_noise

    def __call__(self, image):
        """ Apply a visual effect on the image.

        Args
            image: Image to adjust
        """

        if self.contrast_factor:
            image = adjust_contrast(image, self.contrast_factor)

        if self.brightness_delta:
            image = adjust_brightness(image, self.brightness_delta)

        if self.hue_delta or self.saturation_factor:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if self.hue_delta:
                image = adjust_hue(image, self.hue_delta)
            if self.saturation_factor:
                image = adjust_saturation(image, self.saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        if self.gauss_noise:
            noisy('gauss', image, self.gauss_noise)
        return image

class MyGenerator(tf.keras.utils.Sequence):

    def __init__(
            self,
            batch_size,
            dataset,
            shufle,
            anchor_params,
            test=False,
            shuffle_method="random",
            random_transform=False,
            side_size=800,
            resize_always=False,
            test_only_few=False,
            to_grayscale=False,
            enhance=False
    ):
        self.i = 0
        self.batch_size = int(batch_size)
        self.dataset = dataset
        self.shuffle = shufle
        self.dataset_list = dataset_to_numpy(dataset)
        self.size = len(self.dataset_list)
        self.anchor_params = anchor_params
        self.test = test
        self.transform_parameters = TransformParameters()
        self.side_size = side_size
        self.resize_always = resize_always
        self.test_only_few = test_only_few
        self.to_grayscale = to_grayscale
        self.debug = []
        self.enhance = enhance

        if random_transform:
            self.transform_generator = random_transform_generator(
                                                               # min_rotation=-0.1,
                                                               # max_rotation=0.1,
                                                               # min_translation=(-0.1, -0.1),
                                                               # max_translation=(0.1, 0.1),
                                                               # min_shear=-0.1,
                                                               # max_shear=0.1,
                                                                min_scaling=(0.9, 0.9),
                                                                max_scaling=(1.2, 1.2),
                                                               # flip_x_chance=0.5,
                                                               # flip_y_chance=0.5,
                                                            )

            self.visual_effect_generator = self.random_visual_effect_generator(
                # contrast_range=(1.45, 1.5),
                # brightness_range=(-.15, .25),
                contrast_range=(0.7, 1.5),
                brightness_range=(-.1, .1),
                hue_range=(-0.05, 0.05),
                saturation_range=(0.0, 1.2),
                gauss_noise=(0.0, 0.1)
            )  # todo -- 1.denoise 2. blur/sharpen ;
        else:
            self.transform_generator = None  # random_transform_generator(flip_x_chance=0.5)
            self.visual_effect_generator = None

        if shufle:
            self.group_method = shuffle_method
        else:
            self.group_method = "no"

        self.mix_groups()
        # print("dataset size: {}".format(self.size))
        # with open ("./{}.txt".format(self.size), 'w') as file:
        #
        #     for dat in self.dataset_list:
        #         s = []
        #         s.append(dat['image'].shape[0])
        #         for bbox in dat['bboxes']:
        #             y_r, x_r, h_r, w_r = utils.TLBR_to_center_hw(bbox)
        #             h_r = int(h_r.numpy())
        #             w_r = int(w_r.numpy())
        #             s.append(h_r)
        #             s.append(w_r)
        #         print(*s, file=file)

    def mix_groups(self):
        # determine the order of the images
        order = list(range(self.size))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))  # this seems really usefull # todo
        else:
            print("NOT SHUFFLING")
            pass

        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def img_to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def image_aspect_ratio(self, image_index):
        img = self.dataset_list[image_index]['image']
        width = img.shape[1]
        height = img.shape[0]

        if width != height:
            print("1: {} , 0: {}".format(width, height))

        return float(width) / float(height)

    def on_epoch_end(self):
        self.mix_groups()
        # self.dataset_list = dataset_to_numpy(self.dataset)


    # #############################

    def my_numpy_func(x):
        # x will be a numpy array with the contents of the input to the
        # tf.function
        return np.sinh(x)

    @tf.function # (input_signature=[tf.TensorSpec(None, tf.float32)])
    def compute_targets_numpy_function(self, inp):

        # This func must accept as many arguments as there are tensors in inp
        # accept as many arguments as there are tensors in inp, types match the corresponding tf.Tensor objects

        # The returns numpy.ndarrays must match the number and types defined Tout
        # must return list of numpy.ndarray ( or single ndarray)

        # inp: A list of tf.Tensor objects.
        # Tout: A list or tuple of tensorflow data types or a single tensorflow data type if there is only one,
        # indicating what func returns
        classes, boxes, anchorsTLBR = tf.numpy_function(func=utils.compute_targets, inp=inp, Tout=[tf.float32, tf.float32])
        return classes, boxes

    # tf_function(tf.constant(1.))

    def random_visual_effect_generator(self,
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05),
            gauss_noise=None
    ):

        def _uniform(val_range):
            if val_range is None:
                return None
            """ Uniformly sample from the given range.

            Args
                val_range: A pair of lower and upper bound.
            """
            return np.random.uniform(val_range[0], val_range[1])


        def _check_range(val_range, min_val=None, max_val=None):
            if val_range is None:
                return
            """ Check whether the range is a valid range.
            
            Args
                val_range: A pair of lower and upper bound.
                min_val: Minimal value for the lower bound.
                max_val: Maximal value for the upper bound.
            """
            if val_range[0] > val_range[1]:
                raise ValueError('interval lower bound > upper bound')
            if min_val is not None and val_range[0] < min_val:
                raise ValueError('invalid interval lower bound')
            if max_val is not None and val_range[1] > max_val:
                raise ValueError('invalid interval upper bound')


        """ Generate visual effect parameters uniformly sampled from the given intervals.

        Args
            contrast_factor:   A factor interval for adjusting contrast. Should be between 0 and 3.
            brightness_delta:  An interval between -1 and 1 for the amount added to the pixels.
            hue_delta:         An interval between -1 and 1 for the amount added to the hue channel.
                               The values are rotated if they exceed 180.
            saturation_factor: An interval for the factor multiplying the saturation values of each
                               pixel.
        """
        _check_range(contrast_range, 0)
        _check_range(brightness_range, -1, 1)
        _check_range(hue_range, -1, 1)
        _check_range(saturation_range, 0)

        def _generate():
            while True:
                yield VisualEffect(
                    contrast_factor=_uniform(contrast_range),
                    brightness_delta=_uniform(brightness_range),
                    hue_delta=_uniform(hue_range),
                    saturation_factor=_uniform(saturation_range),
                    gauss_noise=_uniform(gauss_noise)
                )

        return _generate()


    def random_transform_group_entry(self, image, bboxes):
        """ Randomly transforms image and annotation.
        """
        transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

        # apply transformation to image
        image = apply_transform(transform, image, self.transform_parameters)

        # Transform the bounding boxes in the annotations.
        bboxes = bboxes.copy()
        for index in range(bboxes.shape[0]):
            bboxes[index, :] = transform_aabb(transform, bboxes[index, :])
        return image, bboxes

    def get_scale_of_img(self, idx):
        if self.resize_always:
            example = self.dataset_list[idx]
            img = example['image'].numpy()
            img, scale = resize_image(img, self.side_size)
            return img, scale
        return None, 1

    def print_calculated_anchors_for_image(self, img, computed_classes, computed_bboxes, anchorsTLBR):
        computed_bboxes_selection = []
        unprocessed_anchors = []
        unprocessed_boxes = []
        # print(computed_bboxes.shape)
        # exit(33)
        if len(computed_bboxes.shape) == 1:
            computed_bboxes = np.reshape(computed_bboxes, (1, computed_bboxes.shape[0]))
            anchorsTLBR = np.reshape(anchorsTLBR, (1, anchorsTLBR.shape[0]))
            computed_classes = np.reshape(computed_classes, (1, computed_classes.shape[0]))

        for i in range(computed_classes.shape[0]):
            # if computed_classes[i, 0] != 1:
                # r = utils.bbox_from_fast_rcnn(computed_classes[i], computed_bboxes[i])
                # computed_bboxes_selection.append(r)

            if computed_classes[i, 0] != 1:
                unprocessed_anchors.append(anchorsTLBR[i])
                unprocessed_boxes.append(computed_bboxes[i])
            # if len(unprocessed_anchors) > 3:
            #     break

        # RESHAPE && STACK ..
        # computed_bboxes_selection = np.stack(computed_bboxes_selection)
        if len(unprocessed_anchors) > 1:
            unprocessed_anchors = np.stack(unprocessed_anchors)
            unprocessed_boxes = np.stack(unprocessed_boxes)
        else:
            # print(unprocessed_anchors[0].shape)
            unprocessed_anchors = np.reshape(unprocessed_anchors[0], (1, unprocessed_anchors[0].shape[0]))
       # ssert len(unprocessed_anchors.shape) == 3

        print_image_with_rectangles(img, unprocessed_anchors)
        sleep(0.5)

        # unprocessed_anchors = np.reshape(unprocessed_anchors, (1, unprocessed_anchors.shape[0], unprocessed_anchors.shape[1]))
        # unprocessed_boxes = np.reshape(unprocessed_boxes, (1, unprocessed_boxes.shape[0], unprocessed_boxes.shape[1]))
        # #  BBOX from retina representation
        # keras_boxes_selection = (utils.bbox_from_rcnn_keras(unprocessed_anchors, unprocessed_boxes)).numpy()
        # # Reshape and print
        # keras_boxes_selection = np.reshape(keras_boxes_selection, (keras_boxes_selection.shape[1], keras_boxes_selection.shape[2]))
        # print_image_with_rectangles(img, keras_boxes_selection, )

    def __getitem__(self, group_index):



        group = self.groups[group_index]
        examples = [self.dataset_list[idx] for idx in group]

        images_lst = []
        classes_lst = []
        boxes_lst = []

        # if self.size == 4535 or self.size == 1267:  # some debug stuff
        #     self.i += 1
        #     if self.i > 1 and self.i % 500 == 0:
        #         print('{}/{}'.format(self.i, self.size))

        for example in examples:
            img = example['image'].numpy()
            bboxes = example['bboxes'].numpy()
            classes = example['classes'].numpy()

            # print_image_with_rectangles(img, bboxes)

            if self.batch_size > 1 or self.resize_always:  # resize if larger batch ...
                img, scale = resize_image(img, self.side_size)
                bboxes = bboxes / scale

            if self.visual_effect_generator is not None:
                effect = next(self.visual_effect_generator)
                img = effect(image=img)

            if self.enhance:
                from skimage.filters import threshold_yen
                from skimage.exposure import rescale_intensity

                yen_threshold = threshold_yen(img)
                img = rescale_intensity(img, (0, yen_threshold), (0, 1))

            if self.transform_generator is not None:
                img, bboxes = self.random_transform_group_entry(img, bboxes)

            if self.to_grayscale:
                pass
                # img = self.img_to_grayscale(img)
                # img = np.reshape(img, (img.shape[0], img.shape[1], 1))

                # img = adjust_contrast(img, 1.1)

                # img*=256
                # img = img.astype(np.uint8)
                # img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

                # # # #  Create our shapening kernel, it must equal to one eventually
                # # kernel_sharpening = np.array([[-1,-1,-1],
                # #                               [-1, 9,-1],
                # #                               [-1,-1,-1]])
                # # # # # applying the sharpening kernel to the input image & displaying it.
                # # img = cv2.filter2D(img, -1, kernel_sharpening)
                # img/=256.

                    # img = img[:, :, 0]  # only single channel


            max_shape = tf.constant([img.shape[0], img.shape[1], img.shape[2]], dtype=tf.int32)
            # print("image shape {}:".format(max_shape))

            # print_image_with_rectangles(img, bboxes)
            # print(classes)


            # computed_classes, computed_bboxes = self.compute_targets_numpy_function([example['classes'],
            # example['bboxes'], max_shape, self.anchor_params.ratios, self.anchor_params.scales,
            # self.anchor_params.sizes, self.anchor_params.strides, self.anchor_params.levels,
            # self.anchor_params.iou_threshold])

            computed_classes, computed_bboxes, anchorsTLBR = utils.compute_targets(classes,
                                                                      bboxes,
                                                                      max_shape,
                                                                      self.anchor_params.ratios,
                                                                      self.anchor_params.scales,
                                                                      self.anchor_params.anchor_sizes,
                                                                      self.anchor_params.anchor_strides,
                                                                      self.anchor_params.levels,
                                                                      self.anchor_params.iou_threshold)

            tmp=1

            for i in range(computed_classes.shape[0]):
                if computed_classes[i, 0] != 1:
                    tmp += 1

            self.debug.append(tmp)


            if self.test_only_few:
                fig, ax = plt.subplots(1)
                ax.imshow(img)
                plt.show()
                # self.print_calculated_anchors_for_image(img, computed_classes, computed_bboxes, anchorsTLBR)
                # # pass
                # for i in range(computed_bboxes.shape[0]):
                #     # print(computed_bboxes.shape)
                #     if i > 30:
                #         self.print_calculated_anchors_for_image(img, computed_classes[i, : ], computed_bboxes[i, :], anchorsTLBR[i,:])


            images_lst.append(img)
            classes_lst.append(computed_classes)
            boxes_lst.append(computed_bboxes)

        batched_images = np.stack(images_lst)
        batched_bboxes = np.stack(boxes_lst)
        batched_classes = np.stack(classes_lst)
        # print(batched_bboxes.shape)
        # exit(32)

        return batched_images, {"regression_for_loss": batched_bboxes, "classification_for_loss": batched_classes}

    def __len__(self):
        if self.test_only_few:
            if self.size == 10000:  # todo just tmp
                return 5
            if len(self.groups) == 1267:
                return 20

        return len(self.groups)


def dataset_to_numpy(dataset):
    dataset = dataset.map(SVHN.parse)  # parse data into tensor

    list_data = []
    for example in dataset:
        list_data.append(example)

    return list_data




class SVHN:
    TOP, LEFT, BOTTOM, RIGHT = range(4)
    LABELS = 10

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/"

    @staticmethod
    def parse(example):  # jakoby compute input output
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),  # 3 channel floats (h,w,3)
            "classes": tf.io.VarLenFeature(tf.int64),       # list of classes  ()
            "bboxes": tf.io.VarLenFeature(tf.int64)})       # list of bboxes   [x1,y,1,x2,y2]
        example["image"] = tf.image.decode_png(example["image"], channels=3)
        example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
        example["classes"] = tf.sparse.to_dense(example["classes"])
        example["bboxes"] = tf.reshape(tf.sparse.to_dense(example["bboxes"]), [-1, 4])

        # @tf.function # (input_signature=[tf.TensorSpec(None, tf.float32)])
        # def tf_function(classes, bboxes):
        #     return tf.numpy_function(utils.compute_targets, [classes, bboxes], [tf.int64, tf.float64])
        # targets = tf_function(example["classes"], example["bboxes"])
        # targets = utils.compute_targets(example)

        return example  # ["image"], targets

    @staticmethod
    def create_dataset(dataset, batch_size, shuffle=False, augment=False, repeat_augmented=6):
        dataset = dataset.map(SVHN.parse)

        print("creating the dataset")
        if augment:
            if repeat_augmented is not None:
                # print("repeating the dataset {} times".format(repeat_augmented))
                dataset = dataset.repeat(repeat_augmented)  # repeat
            dataset = dataset.shuffle(1000 + 3 * batch_size)  # 10000 + 3 * batch_size [43 batches po 50 images ... 450 training datas ??]
            # dataset = dataset.map(CAGS.preprocess)  # augment

        dataset = dataset.batch(batch_size)

        return dataset


    def __init__(self):
        for dataset in ["train", "dev", "test"]:
            path = "svhn.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

            setattr(self, dataset, tf.data.TFRecordDataset(path))
