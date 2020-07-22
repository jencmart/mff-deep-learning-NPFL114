#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys
import urllib.request
import math

import numpy as np
import tensorflow as tf
import svhn_dataset
import efficient_net
from bboxes_utils import PriorProbability
from svhn_dataset import SVHN
from svhn_dataset import MyGenerator
from visualization import print_image_with_rectangles


def correct_predictions(gold_classes, gold_bboxes, predicted_classes, predicted_bboxes, iou_threshold=0.5):
    if len(gold_classes) != len(predicted_classes):
        return False
    best_iou = - 1

    used = [False] * len(gold_classes)
    for cls, bbox in zip(predicted_classes, predicted_bboxes):
        best = None
        for i in range(len(gold_classes)):
            if used[i] or gold_classes[i] != cls:
                continue
            iou = svhn_dataset.utils.bbox_iou(bbox, gold_bboxes[i])
            if iou >= iou_threshold and (best is None or iou > best_iou):
                best, best_iou = i, iou
        if best is None:
            return False
        used[best] = True
    return True


def visualise_images(gold_data, saved_file, indexes_to_visualise=None):
    if indexes_to_visualise is None:
        indexes_to_visualise = [0]

    i = -1
    printed = 0
    mapped_gold_dataset = gold_data.map(SVHN.parse)
    with open(saved_file, "r", encoding="utf-8-sig") as predictions_file:
        for example in mapped_gold_dataset:
            i += 1
            predictions = [float(value) for value in predictions_file.readline().split()]
            assert len(predictions) % 6 == 0  # label, score, T, L, B, R
            predictions = np.array(predictions, np.float32).reshape([-1, 6])

            if i in indexes_to_visualise:
                print_image_with_rectangles(
                    example["image"].numpy(),
                    example["bboxes"].numpy(),
                    predictions[:, 0].astype(np.int32),  # labels
                    predictions[:, 1],  # scores
                    predictions[:, 2:].astype(np.int32),  # boxes
                    only_predictions=True
                )
                printed += 1
                if printed == len(indexes_to_visualise):
                    break


def straka_metric(gold_dataset, saved_file):
    mapped_gold_dataset = gold_dataset.map(SVHN.parse)
    # Read the predictions
    correct, total = 0, 0
    with open(saved_file, "r", encoding="utf-8-sig") as predictions_file:
        for example in mapped_gold_dataset:
            predictions = [int(value) for value in predictions_file.readline().split()]  # read one line at a time
            assert len(predictions) % 5 == 0

            predictions = np.array(predictions, np.float32).reshape([-1, 5])

            predicted_classes = predictions[:, 0]
            predicted_boxes = predictions[:, 1:]

            correct += correct_predictions(example["classes"].numpy(),
                                           example["bboxes"].numpy(),
                                           predicted_classes,
                                           predicted_boxes)
            total += 1
    print("Correct: {}".format(correct))
    print("Wrong: {}".format(total - correct))
    print("Acc: {:.2f}".format(100 * correct / total))


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

        return result

        # # compute the focal loss - fyzir
        # alpha_factor = tf.keras.backend.ones_like(y_true) * alpha
        # alpha_factor = tf.where(tf.keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        #
        # modulating_factor = tf.where(tf.keras.backend.equal(y_true, 1), 1 - y_pred, y_pred) ** gamma
        # cls_loss = alpha_factor * modulating_factor * ce

        # ###################
        alpha_factor = tf.keras.backend.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.keras.backend.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # ###################

        # compute the normalizer: the number of positive anchors
        # all batches, all claases in image, class 0
        anchor_state = y_true[:, :,
                       0]  # we have one hot encoded 0 as ignore ...  # -1 for ignore, 0 for background, 1 for object
        normalizer = tf.where(tf.keras.backend.not_equal(anchor_state, 1))  # foreground where not 1 on first place...
        normalizer = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
        # true false true false ...
        # 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
        normalizer = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

        return tf.keras.backend.sum(cls_loss) / normalizer  # vydelime poctem class ktere jsou relne na obrazku

    return _focal


# todo --- already fixed
def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression = y_pred
        regression_target = y_true
        anchor_state = tf.math.reduce_sum(y_true, axis=2)

        # filter out "ignore" anchors
        indices = tf.where(tf.keras.backend.not_equal(anchor_state, 0))
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


class AnchorParams:
    def __init__(self,
                 ratios=None,
                 scales=None,
                 anchor_sizes=None,
                 anchor_strides=None,
                 levels=None,
                 ):
        if levels is None:
            levels = [2, 4, 5]
        if anchor_strides is None:
            anchor_strides = [7, 16, 32, ]
        if anchor_sizes is None:
            anchor_sizes = [15, 32, 50]
        if scales is None:
            scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        if ratios is None:
            ratios = [0.5, 1, 2]
        self.ratios = ratios
        self.scales = scales
        self.anchor_sizes = anchor_sizes
        self.anchor_strides = anchor_strides
        self.levels = levels

        self.from_levels = []  # ['P3', 'P4', 'P5', 'P6', 'P7']

        # skip_blocks = ['block5b_project_bn', 'block3a_project_bn', 'block2a_project_bn', 'block1a_project_bn']
        self.skip_blocks = {
            'C3': 'block3b_add',  # C3 cca  28x28  -> P3
            'C4': 'block5c_add',  # C4 cca  14x14  -> P4
            'C5': 'top_activation',  # gratest are, lowest detail # C5 cca  7x7    -> P5
        }

        for i in self.levels:
            self.from_levels.append('P' + str(i))

        self.cnt_anchor_types = len(self.ratios) * len(self.scales)

        self.iou_threshold = 0.5
        self.ratios = tf.constant(self.ratios, dtype=tf.float32)  # float32
        self.scales = tf.constant(self.scales, dtype=tf.float32)  # float32
        self.anchor_sizes = tf.constant(self.anchor_sizes, dtype=tf.int32)  # int
        self.anchor_strides = tf.constant(self.anchor_strides, dtype=tf.float32)  # int
        self.levels = tf.constant(self.levels, dtype=tf.int32)  # int
        self.iou_threshold = tf.constant(self.iou_threshold, dtype=tf.float32)  # float32


class EfficientnetRetina:
    def __init__(self, mode, log_dir, config, anchor_params,
                 side_size=800,
                 resize_always=False,
                 test_only_few=False,
                 filter_config=None,
                 classification_prior_probability=0.01,
                 glorot_kernel_init_heads=True,
                 image_channels=3

                 ):
        assert mode == "training" or "inference"
        self.image_channels = image_channels
        self.classification_prior_prob = classification_prior_probability
        self.glorot_normal = glorot_kernel_init_heads
        self.config = config
        self.val_accuracy = None
        self.mode = mode
        self.base_log_dir = log_dir  # logs
        self.epoch = 0
        self.experiment_name = self.config.name.lower()

        self.pyramid_filters = 256  # filters that go out of pyramid layers
        self.reg_head_filters = 256  # filters in the 4x head
        self.class_head_filters = 256  # filters in the 4x head

        self.feature_filters = 1280
        self.side_size = side_size
        self.resize_always = resize_always
        self.anchor_params = anchor_params
        self.test_only_few = test_only_few
        self.filter_config = filter_config
        # print(self.cnt_anchor_types)
        # print(self.anchor_params.ratios)
        # print(self.anchor_params.scales)
        #
        # exit(50)
        self.model, self.boxed_model = self.build_model(mode=mode)

        # logging function

    @staticmethod
    def log(text, array=None):
        """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
        """
        if array is not None:
            text = text.ljust(25)
            text += ("shape: {:20}  ".format(str(array.shape)))
            if array.size:
                text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
            else:
                text += ("min: {:10}  max: {:10}".format("", ""))
            text += "  {}".format(array.dtype)
        print(text)

    def find_last(self, from_loss):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.base_log_dir))[1]

        key = self.config.name.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.log_dir))
        # Pick last directory
        dir_name = os.path.join(self.base_log_dir, dir_names[-1])

        # Find the checkpoint with lowest validation loss
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("efficientnet_mask"), checkpoints)

        def last_5chars(x):  # 0.222.h5  ... so 22.hp
            return (x[-6:])

        if from_loss:
            reverse = False
        else:
            reverse = True
        checkpoints = sorted(checkpoints, key=last_5chars, reverse=reverse)

        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[0])
        print("using last found checkpoint")
        print(checkpoint)
        return dir_name, checkpoint, checkpoints[-1][-6: -3]

    def load_weights(self, filepath=None, by_name=True, exclude=None, from_loss=True, load_train=False):
        # todo set correct log dirs when model is loaded ....
        if filepath is None:
            print("No filepath, trying to find last checkpoint")
            dir_name, filepath, self.val_accuracy = self.find_last(from_loss=from_loss)
            if not load_train:
                self.set_log_files(dir_name, filepath)

        # load weights from specified path
        if not os.path.exists(filepath):
            raise Exception("Path to the file with weights does not exists")

        self.model.load_weights(filepath, by_name=by_name)

    def set_efficientnet_trainable(self, from_layer, train_bn=False):
        assert from_layer is None or isinstance(from_layer, str) or (
                230 >= from_layer >= 0), "Efficient net trainable layers are between 0 and 230 !"

        # by name ....
        if isinstance(from_layer, str):
            trainable = False
            for layer in self.model.layers:
                if layer.name == from_layer:
                    trainable = True
                if trainable:
                    nam = layer.name
                    if nam.endswith('bn') and train_bn is False:
                        pass
                    layer.trainable = True
                else:
                    layer.trainable = False

        # by depth ...
        elif from_layer:
            for layer in self.model.layers[:from_layer]:  # do not train the bottom
                # log(layer.name)
                layer.trainable = False

            for layer in self.model.layers[from_layer:]:  # top conv + 7a last conv .....
                # log(layer.name)
                nam = layer.name
                if nam.endswith('bn') and train_bn is False:
                    pass
                layer.trainable = True
        else:
            for layer in self.model.layers[:231]:  # do not train the bottom
                # log(layer.name)
                layer.trainable = False

    def compile_model(self, lr, momentum=None):
        optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=0.001)
        #
        self.model.compile(
            # loss=tf.losses.SparseCategoricalCrossentropy(),
            optimizer=optimizer,
            loss={
                'regression_for_loss': smooth_l1(),
                'classification_for_loss': focal(),
                # tfa.losses.SigmoidFocalCrossEntropy()  # focal()  # tfa.losses.SigmoidFocalCrossEntropy()
            },
            # loss=[mask_loss_layer],
            # metrics=[my_custom_metric],
        )  # regresson los = 0.6 , classification loss = 0.015

    # TRAIN
    def train(self,
              train_dataset,
              val_dataset,
              learning_rate,
              epochs,
              augment=True,
              momentum=None,
              repeat_augmented=None,
              finetune_efficient_net_from_layer=None,
              train_bn=False,  # train_bn == False --> BN layers are freezed
              # test_dataset = None,
              random_transform=True,
              to_grayscale=False,
              continue_training=False
              ):  # 225

        assert self.mode == "training", "You need to have model in training mode."

        self.set_log_files()

        if continue_training:
            model.load_weights(from_loss=True, load_train=True)

        # train_generator = MyGenerator(5115000, self.config.batch_size, tfrecords.train, shufle=True)
        # dev_generator = MyGenerator(85078, self.config.batch_size, tfrecords.dev, shufle=True)

        train_generator = MyGenerator(batch_size=self.config.batch_size, dataset=train_dataset, shufle=True,
                                      anchor_params=self.anchor_params, random_transform=random_transform,
                                      resize_always=self.resize_always,
                                      side_size=self.side_size,
                                      to_grayscale=to_grayscale)
        if val_dataset is not None:
            dev_generator = MyGenerator(batch_size=self.config.batch_size, dataset=val_dataset, shufle=True,
                                        anchor_params=self.anchor_params, random_transform=random_transform,
                                        resize_always=self.resize_always,
                                        side_size=self.side_size,
                                        to_grayscale=to_grayscale)
        else:
            dev_generator = None

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):  # logs/
            os.makedirs(self.log_dir)

        # Callbacks # todo -- what to monitor >>>
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True,
                                           write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True,
                                               save_freq='epoch'),
        ]
        # Train
        self.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        self.log("Checkpoint Path: {}".format(self.checkpoint_path))

        self.set_efficientnet_trainable(finetune_efficient_net_from_layer, train_bn=train_bn)
        self.compile_model(learning_rate, momentum)

        print(self.model.summary())

        self.model.fit(
            # train_generator should be
            # inputs = [batch_images, batch_gt_class_ids, batch_gt_masks]
            # outputs = []
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            # steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=dev_generator
            # validation_steps=self.config.VALIDATION_STEPS,
            # max_queue_size=100,
        )

    # BUILD
    def build_model(self, mode, config=None, dynamic_shape=True):

        config = config if config else self.config
        assert mode in ['training', 'inference']

        # 1. Build the backbone - EfficientNet-B0 model
        if self.resize_always:
            input_shape = [None, None, self.image_channels]
        else:
            input_shape = [self.side_size, self.side_size, self.image_channels]

        img_input = tf.keras.layers.Input(shape=input_shape, name="image")
        efficientnet_b0 = self.pretrained_efficientnet_b0(include_top=False,
                                                          input_tensor=img_input)
        efficientnet_b0.trainable = False

        feature_map = efficientnet_b0.output[1]  # 7x7 , 1280
        # for i in efficientnet_b0.outputs:
        #     print(i.name)
        #     print(i.shape)
        # exit(1)

        pyramid_features = self.build_fpn(efficientnet_b0, feature_map)
        reg_class_head = self.build_reg_and_class_heads(pyramid_features)  # pyramid_features

        inputs = [img_input]
        outputs = reg_class_head  # [mask_head]  # class_logits, class_probs,

        model = tf.keras.Model(inputs, outputs, name="EfficientnetRetina")

        boxed_model = self.add_bbox_head(model, from_levels=self.anchor_params.from_levels)

        return model, boxed_model

    def pretrained_efficientnet_b0(self,
                                   include_top,
                                   input_shape=None,
                                   input_tensor=None):
        url = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/models/"
        path = "efficientnet-b0_noisy-student.h5"

        if not os.path.exists(path):
            print("Downloading file {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(url, path), filename=path)

        weights = 'imagenet'
        weights = "efficientnet-b0_noisy-student.h5"

        # return EfficientNetB0(include_top, weights="efficientnet-b0_noisy-student.h5",
        #                       input_shape=[None, None, 3] if dynamic_shape else None)

        return efficient_net.EfficientNet(
            width_coefficient=1.0,
            depth_coefficient=1.0,
            default_resolution=224,
            dropout_rate=0.2,
            model_name='efficientnet-b0',
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,  # tf.keras.layers.Input(shape=input_shape)
            input_shape=input_shape,
            classes=1000
        )

    def add_bbox_head(self, model, from_levels):
        """
        Returns
            A keras.models.Model which takes an image as input and outputs the detections on the image.

            The order is defined as follows:
            ```
            [
                boxes, scores, labels, other[0], other[1], ...
            ]
            ```
        """

        # compute the anchors
        features = [model.get_layer(p_name).output for p_name in from_levels]
        anchors = svhn_dataset.utils.build_anchors(self.anchor_params, features)

        # we expect the anchors, regression and classification values as first output
        regression = model.outputs[0]
        classification = model.outputs[1]

        # apply predicted regression to anchors
        boxes = svhn_dataset.utils.RegressBoxes(name='boxes')([anchors, regression])
        boxes = svhn_dataset.utils.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

        # detections = [boxes, classification]
        # filter detections (apply NMS / score threshold / select top-k)

        detections = svhn_dataset.utils.FilterDetections(
            nms=self.filter_config.nms,
            class_specific_filter=self.filter_config.class_specific_filter,
            name='filtered_detections',
            nms_threshold=self.filter_config.nms_threshold,
            score_threshold=self.filter_config.score_threshold,
            max_detections=self.filter_config.max_detections,  # max detections
            preliminary_remove_bg_class=self.filter_config.preliminary_remove_bg_class,
            parallel_iterations=1
        )([boxes, classification])

        # construct the model
        return tf.keras.models.Model(inputs=model.inputs, outputs=detections, name="bbox-head-output")

    def get_reggression_head(self, corners, cnt_anchor_types):

        input_filters = self.pyramid_filters
        name = "regression_model"

        if self.glorot_normal:
            kernel_init = tf.keras.initializers.GlorotNormal()
        else:
            kernel_init = tf.keras.initializers.GlorotUniform()

        inputs = tf.keras.layers.Input(shape=(None, None, input_filters))
        x = inputs
        for i in range(4):
            x = tf.keras.layers.Conv2D(filters=self.reg_head_filters, kernel_size=3, strides=1, padding='same',
                                       activation='relu',
                                       kernel_initializer=kernel_init,
                                       name='regression_head_{}'.format(i))(x)
        out = tf.keras.layers.Conv2D(cnt_anchor_types * corners, kernel_size=3, strides=1, padding='same',
                                     name='regression_head_last_conv')(x)
        out = tf.keras.layers.Reshape((-1, corners), name='regression_head_resphape')(out)
        return tf.keras.models.Model(inputs=inputs, outputs=out, name=name)

    def get_classification_head(self, cnt_labels, cnt_anchor_types):
        if self.glorot_normal:
            kernel_init = tf.keras.initializers.GlorotNormal()
        else:
            kernel_init = tf.keras.initializers.GlorotUniform()

        input_filters = self.pyramid_filters
        name = "classification_model"

        inputs = tf.keras.layers.Input(shape=(None, None, input_filters))
        x = inputs
        for i in range(4):
            x = tf.keras.layers.Conv2D(filters=self.class_head_filters, kernel_size=3, strides=1, padding='same',
                                       activation='relu',
                                       kernel_initializer=kernel_init,
                                       name='classification_head_{}'.format(i))(x)

        out = tf.keras.layers.Conv2D(filters=cnt_labels * cnt_anchor_types, kernel_size=3, strides=1, padding='same',
                                     bias_initializer=PriorProbability(probability=self.classification_prior_prob),
                                     name='classification_head_last_conv')(x)

        out = tf.keras.layers.Reshape((-1, cnt_labels), name='classification_head_resphape')(out)
        out = tf.keras.layers.Activation('sigmoid', name='classification_head_sigmoid')(out)
        return tf.keras.models.Model(inputs=inputs, outputs=out, name=name)

    # for each pyramid output build the regression and classification heads
    def build_reg_and_class_heads(self, pyramid_feature_list):
        reg_model = self.get_reggression_head(4,
                                              self.anchor_params.cnt_anchor_types)  # out filters ==  corners * anchor_types == 9 * 4 = 26
        class_model = self.get_classification_head(SVHN.LABELS + 1,
                                                   self.anchor_params.cnt_anchor_types)  # SVHN.LABELS + 1
        # + 1 kvuli BG ....

        reg_head = tf.keras.layers.Concatenate(axis=1, name="regression_for_loss")(
            [reg_model(f) for f in pyramid_feature_list])
        class_head = tf.keras.layers.Concatenate(axis=1, name="classification_for_loss")(
            [class_model(f) for f in pyramid_feature_list])

        # print(reg_head.shape)
        # exit(50)

        return [reg_head, class_head]

    def FPNBlock(self, pyramid_filters, stage):
        conv0_name = 'fpn_stage_p{}_pre_conv'.format(stage)
        conv1_name = 'fpn_stage_p{}_conv'.format(stage)
        add_name = 'fpn_stage_p{}_add'.format(stage)
        up_name = 'fpn_stage_p{}_upsampling'.format(stage)

        channels_axis = 3  # 3 if backend.image_data_format() == 'channels_last' else 1

        def wrapper(input_tensor, skip):
            # if input tensor channels not equal to pyramid channels
            # we will not be able to sum input tensor and skip
            # so add extra conv layer to transform it
            # input_filters = backend.int_shape(input_tensor)[channels_axis]
            # if input_filters != pyramid_filters:
            input_tensor = tf.keras.layers.Conv2D(
                filters=pyramid_filters,
                kernel_size=(1, 1),
                kernel_initializer='he_uniform',
                name=conv0_name,
            )(input_tensor)

            skip = tf.keras.layers.Conv2D(
                filters=pyramid_filters,
                kernel_size=(1, 1),
                kernel_initializer='he_uniform',
                name=conv1_name,
            )(skip)

            x = tf.keras.layers.UpSampling2D((2, 2), name=up_name)(input_tensor)
            x = tf.keras.layers.Add(name=add_name)([x, skip])

            return x

        return wrapper

    def __create_pyramid_features(self, C3, C4, C5, feature_size=256):
        """ Creates the FPN layers on top of the backbone features.

        Args
            C3           : Feature stage C3 from the backbone.  cca 7x7
            C4           : Feature stage C4 from the backbone.  cca 14x14
            C5           : Feature stage C5 from the backbone.  cca 28x28
            feature_size : The feature size to use for the resulting feature levels.

        Returns
            A list of feature levels [P3, P4, P5, P6, P7].
        """

        # p7 -> p6
        # p3 -> p4 -> p5

        # p3 p4 p5
        #    p4 p5
        #       p5
        #          p6 p7
        #             p7
        assert len(self.anchor_params.from_levels) > 0, "at least one pyramid output required"

        assert ("P3" not in self.anchor_params.from_levels) or \
               ("P3" in self.anchor_params.from_levels and
                "P4" in self.anchor_params.from_levels and
                "P5" in self.anchor_params.from_levels), "P3 cannot exist without P4 and P5 levels"

        assert ("P4" not in self.anchor_params.from_levels) or \
               ("P4" in self.anchor_params.from_levels and
                "P5" in self.anchor_params.from_levels), "P4 cannot exist without P5 levels"

        assert ("P7" not in self.anchor_params.from_levels) or \
               ("P7" in self.anchor_params.from_levels and
                "P6" in self.anchor_params.from_levels), "P7 cannot exist without P6 levels"

        list_of_pyramid_features = []

        P3 = P4 = P5 = P6 = P7 = None

        if "P6" in self.anchor_params.from_levels:
            # "P6 is obtained via a 3x3 stride-2 conv on C5"
            P6 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)
            if "P7" in self.anchor_params.from_levels:
                # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
                P7 = tf.keras.layers.Activation('relu', name='C6_relu')(P6)
                P7 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

        if "P5" in self.anchor_params.from_levels:
            # upsample C5 to get P5 from the FPN paper
            P5 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(
                C5)  # cca 7x7

            P5_upsampled = svhn_dataset.utils.UpsampleLike(name='P5_upsampled')([P5, C4])
            P5 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(
                P5)  # cca 7x7

        if "P4" in self.anchor_params.from_levels:
            # add P5 elementwise to C4
            P4 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(
                C4)  # cca 14x14
            P4 = tf.keras.layers.Add(name='P4_merged')([P5_upsampled, P4])

            P4_upsampled = svhn_dataset.utils.UpsampleLike(name='P4_upsampled')([P4, C3])
            P4 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(
                P4)  # cca 14x14

        if "P3" in self.anchor_params.from_levels:
            # add P4 elementwise to C3
            P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(
                C3)  # cca 28x28
            P3 = tf.keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
            P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(
                P3)  # cca 28x28

        if P3 is not None:
            list_of_pyramid_features.append(P3)  # 28x28  [ male objekty ]
        if P4 is not None:
            list_of_pyramid_features.append(P4)  # 14x14  [ vetsi objekty ]
        if P5 is not None:
            list_of_pyramid_features.append(P5)  # 7x7    [ velke objekty ]
        if P6 is not None:
            list_of_pyramid_features.append(P6)  # 3x3    [ hodne velke objekty ]
        if P7 is not None:
            list_of_pyramid_features.append(P7)  # 1x1    [ nejvetsi objekty ]

        #
        # print("sizes")
        # print(P3.shape)
        # print(P4.shape)
        # print(P5.shape)
        # print(P6.shape)
        # print(P7.shape)
        return list_of_pyramid_features  # [P3, P4, P5, { P6, P7 } ]

        # TEST

    def test(self, test_dataset, random_transform, to_grayscale):
        # note --- testing / detecting does not requre compile, you stupid bitch :-)
        r = []

        # assert self.mode == "inference", "Create model in inference mode in order to test!"

        # test_dataset_generator = CAGS.create_dataset(test_dataset, args.batch_size, augment=False)

        test_dataset_generator = MyGenerator(batch_size=self.config.batch_size, dataset=test_dataset, shufle=False,
                                             anchor_params=self.anchor_params, test=True,
                                             side_size=self.side_size,
                                             random_transform=random_transform,
                                             resize_always=self.resize_always,
                                             test_only_few=self.test_only_few,
                                             to_grayscale=to_grayscale)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):  # logs/
            os.makedirs(self.log_dir)

        self.log("\nStarting testing\n")

        results = self.boxed_model.predict(test_dataset_generator)

        # results = self.manually_filter_results(results)

        box_results, scores_results, labels_results = results
        # print(labels_results)
        # exit(33)
        i = 0
        assert test_dataset_generator.__len__() == box_results.shape[0], "not the same number of the results as the input images"
        for boxes, scores, labels in zip(box_results, scores_results, labels_results):
            img, scale = test_dataset_generator.get_scale_of_img(i)
            # print_image_with_rectangles(img, boxes)
            boxes = boxes * scale  # resize boxes back to correct size
            boxes = boxes.astype(np.int32)

            # filter out '-1' padding
            boxes = boxes[np.where(labels > -1)]
            scores = scores[np.where(labels > -1)]
            labels = labels[np.where(labels > -1)]
            if labels.shape[0] > 0:
                r.append((labels, scores, boxes))
            else:
                print("this result is empty")
                labels = [0]
                scores = [0.0]
                boxes = [[0,0,1,1]]
                r.append((labels, scores, boxes))
            i += 1

        return r

    def build_fpn(
            self,
            backbone,
            x_out,
    ):
        # skip_connection_layers = self.anchor_params.skip_blocks

        #         c3 = skips[0]  # 28  C3
        #         c4 = skips[1]  # 14  C4
        #         c5 = skips[2]  # 7   C5

        # skip_connection_layers  C5, C4, C3
        # building decoder blocks with skip connections
        # skips = ( [backbone.get_layer(name=i).output if isinstance(i, str) else backbone.get_layer(index=i).output for i in skip_connection_layers])

        c5 = backbone.get_layer(name=self.anchor_params.skip_blocks['C5']).output  # 7   C5
        c4 = backbone.get_layer(name=self.anchor_params.skip_blocks['C4']).output  # 14  C4
        c3 = backbone.get_layer(name=self.anchor_params.skip_blocks['C3']).output  # 28  C3

        # # print(skips[0])
        # # exit(45)
        # # build FPN pyramid
        # c5 = skips[2]
        # c4 = skips[1]
        # c3 = skips[0]

        return self.__create_pyramid_features(c3, c4, c5)

        # p5 = self.FPNBlock(pyramid_filters, stage=5)(x, skips[0])  # skips[0] = 14 x 14 c5
        # p4 = self.FPNBlock(pyramid_filters, stage=4)(p5, skips[1])  # 28 x 28            C4
        # p3 = self.FPNBlock(pyramid_filters, stage=3)(p4, skips[2])  # 56 x 56            C3
        # p2 = self.FPNBlock(pyramid_filters, stage=2)(p3, skips[3])  # 112 x 112          C2
        #
        # # add segmentation head to each
        # s5 = tf.keras.layers.Conv2D(pyramid_filters, kernel_size=3, strides=1, padding='same', name='P5')(p5)
        # s4 = tf.keras.layers.Conv2D(pyramid_filters, kernel_size=3, strides=1, padding='same', name='P4')(p4)
        # s3 = tf.keras.layers.Conv2D(pyramid_filters, kernel_size=3, strides=1, padding='same', name='P3')(p3)
        # s2 = tf.keras.layers.Conv2D(pyramid_filters, kernel_size=3, strides=1, padding='same', name='P2')(p2)
        # s6 = tf.keras.layers.Conv2D(pyramid_filters, kernel_size=3, strides=2, padding='same', name='P6')(s5)
        #
        # s7 = tf.keras.layers.Activation('relu', name='C6_relu')(s6)
        # s7 = tf.keras.layers.Conv2D(pyramid_filters, kernel_size=3, strides=2, padding='same', name='P7')(
        #     s7)  # upsampling to same resolution
        #
        # return [s3, s4, s5, s6, s7]

    def set_log_files(self, log_dir=None, checkpoint_path=None):

        if log_dir is not None and checkpoint_path is not None:
            self.log_dir = log_dir
            self.checkpoint_path = checkpoint_path
            return

        now = datetime.datetime.now()
        self.log_dir = os.path.join(self.base_log_dir,
                                    "{}{:%Y%m%dT%H%M}".format(self.experiment_name, now))  # logs/exp_5_24.12.2020
        # todo -- val iou acc
        base_checkpoint_path = "efficientnet_mask{}".format(self.experiment_name) + ".{epoch:02d}-{val_loss:.3f}.h5"

        self.checkpoint_path = os.path.join(self.log_dir, base_checkpoint_path)

    def _filter_detections(self, scores, labels, boxes, score_threshold=0.05, max_detections=5, nms_threshold=0.5,
                           nms=True):
        # threshold based on score
        indices = tf.where(tf.keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes = tf.gather_nd(boxes, indices)
            filtered_scores = tf.keras.backend.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(filtered_boxes,
                                                       filtered_scores,
                                                       max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = tf.keras.backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = tf.keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    def manually_filter_results(self, results, max_detections=5):
        unfil_boxes, unfil_class = results
        min_score = 0.05

        boxes = []
        classifications = []
        for box, scores in zip(unfil_boxes, unfil_class):
            if np.argmax(scores) > 0 and np.max(scores) > min_score:
                boxes.append(box)
                classifications.append(scores)
        boxes = np.stack(boxes)
        classifications = np.stack(classifications)  # todo maybe to tensor ...

        scores = np.max(classifications, axis=1)
        labels = np.argmax(classifications, axis=1)

        indices = self._filter_detections(scores, labels, boxes, max_detections=max_detections)

        # select top k
        scores = tf.gather_nd(classifications, indices)
        labels = indices[:, 1]
        scores, top_indices = tf.nn.top_k(scores,
                                          k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))

        indices = tf.keras.backend.gather(indices[:, 0], top_indices)
        boxes = tf.keras.backend.gather(boxes, indices)
        labels = tf.keras.backend.gather(labels, top_indices)

        return [boxes, scores, labels]


def save_predictions(results, path, save_scores=False):
    print(path)
    with open(path, "w", encoding="utf-8") as out_file:
        for prediction in results:
            # boxes, scores, labels
            predicted_classes, predicted_scores, predicted_bboxes = prediction
            output = []
            for label, score, bbox in zip(predicted_classes, predicted_scores, predicted_bboxes):
                output.append(label)
                if save_scores:
                    output.append(score)
                output.extend(bbox)
            print(*output, file=out_file)


if __name__ == "__main__":

    # 0.00001 for 200 epochs.
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

    parser.add_argument("--name", default="test", help="name of experiment")
    parser.add_argument("--momentum", default=None, help="if none Adam is used, othervise RMSProp")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--repeat_augmented", default=1, type=int, help="Repeat dataset n times before augmenting")
    parser.add_argument("--finetune_efficient_net_from_layer", default=None, type=str,
                        help="None or 225 to finetune efficient net")
    parser.add_argument("--continue_training", default=False, type=bool, help="Contine traning using the last weights")

    parser.add_argument("--random_tranform", default=False, type=bool, help="Random transform datasets")
    parser.add_argument("--to_grayscale", default=False, type=bool, help="Random transform datasets")

    parser.add_argument("--do_train", default=False, type=bool, help="Train or not ...")
    parser.add_argument("--do_test_dev", default=False, type=bool, help="Test dev dataset or not ...")
    parser.add_argument("--do_test_test", default=False, type=bool, help="Test test dataset or not ...")
    parser.add_argument("--do_straka_dev", default=False, type=bool, help="Test dev dataset or not ...")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    tfrecords = SVHN()

    ROOT_DIR = "./"
    # Directory to save logs and trained model
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

    test_only_few = True  # todo -- only for debug !!! otherwise set to false
    # todo ---- params what to do
    what_to_do = {'train': args.do_train,
                  'test_dev': args.do_test_dev,
                  'test_test': args.do_test_test,
                  'straka_dev': args.do_straka_dev,
                  'visualise': False}
    what_to_do['visualise'] = True

    if what_to_do['test_dev'] or what_to_do['test_test']:
        test_only_few = False

    random_transform = args.random_tranform
    to_grayscale = args.to_grayscale
    if to_grayscale:
        channels = 1
    else:
        channels = 3

    # todo --- change these parameters please !!! -- try more ratios etc ...
    # anchor_params = AnchorParams(
    #     levels=[5, 6, 7],
    #     anchor_sizes=[50, 80, 130],
    #     anchor_strides=[32, 64, 128],  # 4x mensi strides to serou
    #     ratios=[0.5],  # 03. 05. 07
    #     scales=[2 ** 0]  # 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)  #  2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)
    # )

    side_size = 64
    resize_always = True # jit jeste pred level 3
    anchor_params = AnchorParams(
        levels=[3, 4, 5],
        # 4, 5 # 6, 7 ## pocitej s tim, ze na danem levelu mame male H,V takze celkovy pocet anchors je omezen !!!
        # level 3 --> 8,  4 --> 4,   5 --> 2
        #             64       16        4
        # pocet anchors == shape na levelu ^2 == 100 anchors
        anchor_sizes=[16, 32, 50],  # 40, 60   #  80, 130
        anchor_strides=[8, 16, 32],  # 4x mensi strides to serou
        ratios=[0.3, 0.5, 0.7],  # 0.8
        scales=[2 ** 1.2, 2 ** 1, 2 ** (2.0 / 3.0)]
    )

    # pro kazdy pixel na kazdem levelu mame centrum anchor pointu
    shapes = []
    for x in anchor_params.levels:
        shapes.append((side_size + 2 ** x - 1) // (2 ** x))
    shapes = np.array(shapes)
    cnt_anchor_positions = np.sum(shapes * shapes)
    anchor_types = len(anchor_params.ratios) * len(anchor_params.scales)
    cnt_anchors = cnt_anchor_positions * anchor_types
    cnt_anchors = int(cnt_anchors)

    print("cnt anchors: {}".format(cnt_anchors))
    classification_prior_probability = 30 / cnt_anchors  # we have on average 30 positive anchors per image
    print("prior probability anchors: {}".format(classification_prior_probability))


    # todo -- filtering --- can changed later ....
    class FilterConfig:
        def __init__(self,
                     nms=True,
                     class_specific_filter=False,
                     nms_threshold=0.05,    # 0.05 seems to do the best job
                     score_threshold=0.15,  # mezi 0.1 a 0.2
                     max_detections=4,
                     preliminary_remove_bg_class='zero-score'  # 'zero-score', 'remove-rows', 'no' # kydz dam zero score ... a zaroven threshold 0.1 ... 'tezke'
                     ):
            self.nms = nms
            self.class_specific_filter = class_specific_filter
            self.nms_threshold = nms_threshold
            self.score_threshold = score_threshold
            self.max_detections = max_detections
            self.preliminary_remove_bg_class = preliminary_remove_bg_class



    # Correct: 109 (8.6%) nms TRUE, specific fALSE, nms 0.5,  score 0.1 , max detect 4 , zero score
    # Correct: 208 (16%)  nms TRUE, specific fALSE, nms 0.4,  score 0.1 , max detect 4 , zero score
    # Correct: 365 (28%)  nms TRUE, specific fALSE, nms 0.3,  score 0.1 , max detect 4 , zero score
    # Correct: 488 (38%)  nms TRUE, specific fALSE, nms 0.2,  score 0.1 , max detect 4 , zero score
    # Correct: 532 (42%)  nms TRUE, specific fALSE, nms 0.1,  score 0.1 , max detect 4 , zero score
    # Correct: 500 (40%)  nms TRUE, specific fALSE, nms 0.05, score 0.1 , max detect 4 , zero score
    # Correct: 575 (45%)  nms TRUE, specific fALSE, nms 0.05, score 0.15, max detect 4 , zero score
    # Correct: 549  43.33  --- same ---- random transform ----
    # Not transforming 44.44 -- no transform ---

    filter_config = FilterConfig()

    #  I have trained your model correctly on my dataset. And now I want to make some changes,so
    #  I delete P3 and P4 layers since the object in my dataset is large.

    if what_to_do['train']:
        model = EfficientnetRetina(mode="training", log_dir=LOG_DIR, config=args, anchor_params=anchor_params,
                                   side_size=side_size,
                                   resize_always=resize_always,
                                   filter_config=filter_config,
                                   classification_prior_probability=classification_prior_probability,
                                   image_channels=channels)

        model.train(train_dataset=tfrecords.train,
                    val_dataset=tfrecords.dev,
                    learning_rate=args.lr,
                    momentum=args.momentum,
                    epochs=args.epochs,
                    repeat_augmented=args.repeat_augmented,
                    finetune_efficient_net_from_layer=args.finetune_efficient_net_from_layer,
                    random_transform=random_transform,
                    continue_training=args.continue_training,
                    to_grayscale=to_grayscale
                    )

    if what_to_do['test_dev']:
        # Load the model
        model = EfficientnetRetina(mode="inference", log_dir=LOG_DIR, config=args, anchor_params=anchor_params,
                                   side_size=side_size,
                                   resize_always=resize_always,
                                   test_only_few=test_only_few,
                                   filter_config=filter_config,
                                   classification_prior_probability=classification_prior_probability,
                                   image_channels=channels)
        model.load_weights(from_loss=True)

        # Create results directory
        results_path = os.path.join(ROOT_DIR, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Test and save DEV
        results = model.test(tfrecords.dev,
                             random_transform=random_transform,
                             to_grayscale=to_grayscale)
        save_predictions(results, os.path.join(results_path, "svhn_classification_dev.txt"))

        save_predictions(results, os.path.join(results_path, "svhn_classification_dev_scores.txt"), save_scores=True)

    if what_to_do['test_test']:
        # Load the model
        model = EfficientnetRetina(mode="inference",
                                   log_dir=LOG_DIR,
                                   config=args,
                                   anchor_params=anchor_params,
                                   side_size=side_size,
                                   resize_always=resize_always,
                                   test_only_few=test_only_few,
                                   filter_config=filter_config,
                                   classification_prior_probability=classification_prior_probability,
                                   image_channels=channels)
        model.load_weights(from_loss=True)

        # Create results directory
        results_path = os.path.join(ROOT_DIR, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # Test and save TEST
        results = model.test(tfrecords.test,
                             random_transform=random_transform,
                             to_grayscale=to_grayscale)
        save_predictions(results, os.path.join(results_path, "svhn_classification.txt"))

    if what_to_do['visualise']:

        # Create results directory
        results_path = os.path.join(ROOT_DIR, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        visualise_images(tfrecords.dev,
                         os.path.join(results_path, "svhn_classification_dev_scores.txt"),
                         indexes_to_visualise=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    if what_to_do['straka_dev']:
        # Create results directory
        results_path = os.path.join(ROOT_DIR, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        straka_metric(tfrecords.dev, os.path.join(results_path, "svhn_classification_dev.txt"))