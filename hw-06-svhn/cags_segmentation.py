#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from cags_dataset import CAGS
import efficient_net
import urllib.request
import matplotlib.pyplot as plt

# inspired by https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py


def mask_loss_layer(y_true, y_pred): # input_gt_class_ids
    # y_true = tf.image.resize(y_true, (56, 56))  # [batch, height, width]

    # y_true = tf.reshape(y_true, (-1, 224, 224))  # bot needed or both not needed
    # y_pred = tf.reshape(y_pred, (-1, 224, 224))

    loss = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
    loss = tf.math.reduce_mean(loss)
    return loss


def iou_acc(y_true, y_pred):

    # todo --- y_pred ma velikost 14 x 14 ... a to se musi opravit zde
    # y_pred = tf.image.resize(y_pred, (224, 224))

    y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, CAGS.H * CAGS.W])
    y_pred_mask = tf.reshape(tf.math.round(y_pred) == 1, [-1, CAGS.H * CAGS.W])

    intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
    union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

    intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
    union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

    iou = tf.where(union == 0, 1., intersection / union)
    iou = tf.reduce_mean(iou)
    return iou




# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------
def Conv2dBn(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    backend, layers, models, keras_utils = None, None, None, None # get_submodules_from_kwargs(kwargs)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 # if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor):

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = tf.keras.layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper

def Conv3x3BnReLU(filters, use_batchnorm, name=None):

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name
        )(input_tensor)

    return wrapper


def DoubleConv3x3BnReLU(filters, use_batchnorm, name=None):
    name1, name2 = None, None
    if name is not None:
        name1 = name + 'a'
        name2 = name + 'b'

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name1)(input_tensor)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=name2)(x)
        return x

    return wrapper


def FPNBlock(pyramid_filters, stage):
    conv0_name = 'fpn_stage_p{}_pre_conv'.format(stage)
    conv1_name = 'fpn_stage_p{}_conv'.format(stage)
    add_name = 'fpn_stage_p{}_add'.format(stage)
    up_name = 'fpn_stage_p{}_upsampling'.format(stage)

    channels_axis = 3  # 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip):
        # if input tensor channels not equal to pyramid channels
        # we will not be able to sum input tensor and skip
        # so add extra conv layer to transform it
        #input_filters = backend.int_shape(input_tensor)[channels_axis]
        #if input_filters != pyramid_filters:
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

def build_fpn(
        backbone,
        x_out,
        skip_connection_layers,
        pyramid_filters=256,
        segmentation_filters=128,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        aggregation='sum',
        dropout=None,
):
    # input_ = backbone.input
    x = x_out

    # building decoder blocks with skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str) else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # print(skips[0])
    # exit(45)
    # build FPN pyramid
    p5 = FPNBlock(pyramid_filters, stage=5)(x, skips[0])   # skips[0] = 14 x 14
    p4 = FPNBlock(pyramid_filters, stage=4)(p5, skips[1])  # 28 x 28
    p3 = FPNBlock(pyramid_filters, stage=3)(p4, skips[2])  # 56 x 56
    p2 = FPNBlock(pyramid_filters, stage=2)(p3, skips[3])  # 112 x 112

    # add segmentation head to each
    s5 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, use_batchnorm, name='segm_stage2')(p2)

    # upsampling to same resolution
    s5 = tf.keras.layers.UpSampling2D((8, 8), interpolation='nearest', name='upsampling_stage5')(s5)  # 112 x 112 <- 14 x 14
    s4 = tf.keras.layers.UpSampling2D((4, 4), interpolation='nearest', name='upsampling_stage4')(s4)  # 112 x 112 <- 28 x 28
    s3 = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest', name='upsampling_stage3')(s3)  # 112 x 112 <- 56 x 56

    # aggregating results
    if aggregation == 'sum':
        x = tf.keras.layers.Add(name='aggregation_sum')([s2, s3, s4, s5])
    elif aggregation == 'concat':
        concat_axis = 3  # if backend.image_data_format() == 'channels_last' else 1
        x = tf.keras.layers.Concatenate(axis=concat_axis, name='aggregation_concat')([s2, s3, s4, s5])
    else:
        raise ValueError('Aggregation parameter should be in ("sum", "concat"), '
                         'got {}'.format(aggregation))

    if dropout:
        x = tf.keras.layers.SpatialDropout2D(dropout, name='pyramid_dropout')(x)

    # final stage
    x = Conv3x3BnReLU(segmentation_filters, use_batchnorm, name='final_stage')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='final_upsampling')(x)  # 224 x 224

    # model head (define number of output classes)
    x = tf.keras.layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv',
    )(x)
    x = tf.keras.layers.Activation(activation, name=activation)(x)

    return x
    # create keras model instance
    # model = models.Model(input_, x)

    # return model


class EfficientMaskNet:

    def __init__(self, mode, log_dir, config):
        assert mode == "training" or "inference"
        self.config = config
        self.val_accuracy = None
        self.mode = mode
        self.base_log_dir = log_dir  # logs
        self.epoch = 0
        self.experiment_name = self.config.name.lower()

        self.model = self.build_model(mode=mode)


    def set_log_files(self, log_dir=None, checkpoint_path=None):

        if log_dir is not None and checkpoint_path is not None:
            self.log_dir = log_dir
            self.checkpoint_path = checkpoint_path
            return

        now = datetime.datetime.now()
        self.log_dir = os.path.join(self.base_log_dir,
                                    "{}{:%Y%m%dT%H%M}".format(self.experiment_name, now))  # logs/exp_5_24.12.2020
        base_checkpoint_path = "efficientnet_mask{}".format(self.experiment_name) + ".{epoch:02d}-{val_iou_acc:.3f}.h5"
        self.checkpoint_path = os.path.join(self.log_dir,
                                            base_checkpoint_path)  # logs/exp_5_24.12.2020/efficient_mask.h5_

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

    def find_last(self):
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

        def last_5chars(x):  # 0.22.h5  ... so 22.hp
            return (x[-6:])

        checkpoints = sorted(checkpoints, key=last_5chars)

        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1]) # 0 for loss ; -1 for acc ... todo
        print("using last found checkpoint")
        print(checkpoint)
        return dir_name, checkpoint, checkpoints[-1][-6 : -3]

    def load_weights(self, filepath=None, by_name=True, exclude=None):

        if filepath is None:
            print("No filepath, trying to find last checkpoint")
            dir_name, filepath, self.val_accuracy = self.find_last()
            self.set_log_files(dir_name, filepath)

        # load weights from specified path
        if not os.path.exists(filepath):
            raise Exception("Path to the file with weights does not exists")

        self.model.load_weights(filepath, by_name=by_name)

        # todo set correct log dirs when model is loaded ....

    def set_efficientnet_trainable(self, from_layer):
        assert from_layer is None or (230 >= from_layer >= 0) or isinstance(from_layer, str), "Efficient net trainable layers are between 0 and 230 !"

        # by name ....
        if isinstance(from_layer, str):
            trainable = False
            for layer in self.model.layers:
                if layer.name == from_layer:
                    trainable = True
                if trainable:
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
                layer.trainable = True
        else:
            for layer in self.model.layers[:231]:  # do not train the bottom
                # log(layer.name)
                layer.trainable = False

    # TEST
    def test(self, test_dataset):
        # note --- testing / detecting does not requre compile, you stupid bitch :-)
        results = []

        assert self.mode == "inference", "Create model in inference mode in order to test!"

        test_dataset_generator = CAGS.create_dataset(test_dataset, args.batch_size, augment=False)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):  # logs/
            os.makedirs(self.log_dir)

        self.log("\nStarting testing\n")

        mask_head = self.model.predict(test_dataset_generator)
        # print("LETS DO IT")

        # todo single head -- cele toto zakomentovano

        result = mask_head
        for i, _ in enumerate(mask_head):
            full_sized_mask = result[i] #  tf.image.resize(result[i], (224, 224))  # upscale the mask

            results.append({
                "mask": full_sized_mask
            })

        return results

    # TRAIN
    def train(self,
              train_dataset,
              val_dataset,
              learning_rate,
              epochs,
              augment=True,
              momentum=None,
              repeat_augmented=None,
              finetune_efficient_net_from_layer=None):  # 225

        assert self.mode == "training", "You need to have model in training mode."

        self.set_log_files()

        train_dataset_generator = CAGS.create_dataset(train_dataset, self.config.batch_size, shuffle=True,
                                                      augment=augment, repeat_augmented=repeat_augmented)
        val_dataset_generator = CAGS.create_dataset(val_dataset, self.config.batch_size, shuffle=True, augment=False)

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

        self.set_efficientnet_trainable(finetune_efficient_net_from_layer)
        self.compile_model(learning_rate, momentum)

        try:
            self.model.fit(
                # train_generator should be
                # inputs = [batch_images, batch_gt_class_ids, batch_gt_masks]
                # outputs = []
                train_dataset_generator,
                initial_epoch=self.epoch,
                epochs=epochs,
                # steps_per_epoch=self.config.STEPS_PER_EPOCH,
                callbacks=callbacks,
                validation_data=val_dataset_generator
                # validation_steps=self.config.VALIDATION_STEPS,
                # max_queue_size=100,

            )

        except KeyboardInterrupt:
            print('Interrupted')
            return

        # self.epoch = max(self.epoch, epochs)

    # BUILD
    def build_model(self, mode, config=None, dynamic_shape=False):
        config = config if config else self.config
        assert mode in ['training', 'inference']

        # 1. Build the backbone - EfficientNet-B0 model
        default_resolution = 224
        input_shape = [None, None, 3] if dynamic_shape else [default_resolution, default_resolution, 3]
        img_input = tf.keras.layers.Input(shape=input_shape, name="image")
        efficientnet_b0 = self.pretrained_efficientnet_b0(include_top=False,
                                                          input_tensor=img_input)
        efficientnet_b0.trainable = False

        feature_map = efficientnet_b0.output[1]  # 7x7 , 1280

        # for layer in efficientnet_b0.layers:
        #     print("name: {}   shape: {}".format(layer.name, layer.output_shape))

        skip_blocks = ['block5b_project_bn', 'block3a_project_bn', 'block2a_project_bn', 'block1a_project_bn']

        mask_head = build_fpn(efficientnet_b0, feature_map, skip_blocks)

        # mask_head = self.build_mask_head(feature_map, config.train_bn, CAGS.num_classes_types,
        #                                  num_convs=self.config.num_convs)
        #
        # mask_head = tf.image.resize(mask_head, (224, 224))

        inputs = [img_input]
        outputs = [mask_head]  # class_logits, class_probs,


        model = tf.keras.Model(inputs, outputs, name="efficientnet_mask")

        return model

    # def build_classification_head(self, gap_feature_map, train_bn, dropout_rate):
    #     if dropout_rate > 0:
    #         gap_feature_map = tf.keras.layers.Dropout(dropout_rate, name="dropout_out")(gap_feature_map)
    #     class_logits = tf.keras.layers.Dense(CAGS.num_classes_types, name="class_logits")(gap_feature_map)  # 2
    #     class_probs = tf.keras.activations.softmax(class_logits)  # name="class_out"
    #
    #     return class_logits, class_probs

    def build_mask_head(self, feature_map, train_bn, num_classes, num_convs=4):
        x = feature_map


        x = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), name="mask_deconv00")(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.BatchNormalization(name="mask_bn00", trainable=train_bn)(x)

        for i in range(num_convs):
            # Layer 1: conv -> bn -> relu
            x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                       name="mask_conv{}".format(i))(x)
            x = tfa.layers.GroupNormalization(groups=8, axis=3)(x)
            # x = tf.keras.layers.Ba(name="mask_bn{}".format(i), trainable=train_bn)(x)
            x = tf.keras.activations.relu(x)

        # de-conv -> relu
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), name="mask_deconv")(x)
        x = tf.keras.activations.relu(x)

        x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), name="mask_deconv2")(x)
        x = tf.keras.activations.relu(x)

        # conv -> sigmoid
        x = tf.keras.layers.Conv2D(num_classes, (1, 1), strides=(1, 1), name="mask_out")(x)
        x = tf.keras.activations.sigmoid(x)

        # print(x.shape)

        return x

    def compile_model(self, lr, momentum=None):

        if momentum:
            optimizer = tf.optimizers.RMSprop(learning_rate=lr, momentum=momentum)  # tf.optimizers.RMSprop(lr=2e-5)
        else:
            optimizer = tf.optimizers.Adam(learning_rate=lr)

        # loss_names = ["mask_loss"]  # "class_loss",  # todo single head
        #
        # # load layers representing the loss
        # for name in loss_names:
        #     layer = self.model.get_layer(name)
        #
        #     loss = tf.reduce_mean(input_tensor=layer.output, keepdims=True)
        #
        #     self.model.add_loss(losses=loss)

        # SparseCategoricalCrossentropy == 0,1,2...
        self.model.compile(
            # loss=tf.losses.SparseCategoricalCrossentropy(),
            optimizer=optimizer,
            #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #metrics=['accuracy']

            # loss=[None] * len(self.model.outputs),
            loss=[mask_loss_layer],
            metrics=[iou_acc],
            # metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")], # todo metric will be added as callback
        )
        # print(self.model.summary())

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def manually_validate(predicition_path, dataset="dev"):
    # Load the gold data
    # gold_masks = getattr(CAGS(), dataset).map(CAGS.parse).map(lambda example: example["mask"])
    gold_masks = getattr(CAGS(), dataset).map(CAGS.parse).map(lambda img, mask: mask)

    # Create the metric
    iou = CAGSMaskIoU()

    # Read the predictions
    with open(predicition_path, "r", encoding="utf-8-sig") as predictions_file:
        for gold_mask in gold_masks:
            predicted_runs = [int(run) for run in predictions_file.readline().split()]
            # print(sum(predicted_runs))
            # print( CAGS.H * CAGS.W)
            assert sum(predicted_runs) == CAGS.H * CAGS.W

            predicted_mask = np.zeros([CAGS.H * CAGS.W], np.int32)
            offset = 0
            for i, run in enumerate(predicted_runs):
                predicted_mask[offset:offset + run] = i % 2
                offset += run

            iou(gold_mask, predicted_mask)

    print("{:.2f}".format(100 * iou.result()))
    return iou.result()


class CAGSMaskIoU(tf.metrics.Mean):
    """CAGSMaskIoU computes IoU for CAGS dataset masks predicted by binary classification"""

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, CAGS.H * CAGS.W])
        y_pred_mask = tf.reshape(tf.math.round(y_pred) == 1, [-1, CAGS.H * CAGS.W])

        intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
        union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

        intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
        union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

        iou = tf.where(union == 0, 1., intersection / union)
        return super().update_state(iou, sample_weight)


def save_results_to_file(file_path, results):
    with open(file_path, "w", encoding="utf-8") as out_file:
        for result in results:
            mask = result["mask"]
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=out_file)

    print("output saved")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

    parser.add_argument("--finetune_efficient_net_from_layer", default=None, type=int,
                        help="None or 225 to finetune efficient net")

    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="dropout rate.")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate.")
    parser.add_argument("--from_last_checkpoint", default=False, help="Continue training from last checkpoint...")
    parser.add_argument("--train_bn", default=False, help="Batch normalization during training")
    parser.add_argument("--name", default="test", help="name of experiment")
    parser.add_argument("--momentum", default=None, help="if none Adam is used, othervise RMSProp")
    parser.add_argument("--repeat_augmented", default=1, type=int, help="Repeat dataset n times before augmenting")
    parser.add_argument("--num_convs", default=1, type=int, help="Number of convolutions in mask head")
    parser.add_argument("--train_both", default=False, type=bool, help="Train on both train + validation data")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    ROOT_DIR = "./"
    # Directory to save logs and trained model
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

    # Load the data
    cags = CAGS()

    train = True
    if train:
        model = EfficientMaskNet(mode="training", log_dir=LOG_DIR, config=args)

        # load last checkpoint or not
        if args.from_last_checkpoint:
            model.load_last_checkpoint()

        train_data = cags.train
        if args.train_both:
            train_data = cags.both
            print("using BOTH DATASETS FOR TRAINING")

        model.train(train_dataset=train_data,
                    val_dataset=cags.dev,
                    learning_rate=args.lr,
                    momentum=args.momentum,
                    epochs=args.epochs,
                    repeat_augmented=args.repeat_augmented,
                    finetune_efficient_net_from_layer=args.finetune_efficient_net_from_layer  # 225
                    )

        # testing == inference
        model = EfficientMaskNet(mode="inference", log_dir=LOG_DIR, config=args)
        model.load_weights()

        # save test results
        test_results_path = os.path.join(LOG_DIR, "cags_segmentation_{}.txt".format(model.val_accuracy))
        results = model.test(cags.test)
        save_results_to_file(test_results_path, results)

        # save dev results
        dev_results_path = os.path.join(LOG_DIR, "cags_segmentation_dev.txt")
        results_dev = model.test(cags.dev)
        save_results_to_file(dev_results_path, results_dev)

        # validate dev dataset
        manually_validate(dev_results_path, dataset="dev")

    else:
        # testing == inference
        model = EfficientMaskNet(mode="inference", log_dir=LOG_DIR, config=args)
        model.load_weights()

        # save test results
        results = model.test(cags.test)
        test_results_path = os.path.join(LOG_DIR, "cags_segmentation.txt")
        save_results_to_file(test_results_path, results)

        # save dev results
        dev_results_path = os.path.join(LOG_DIR, "cags_segmentation_dev.txt")
        results_dev = model.test(cags.dev)
        save_results_to_file(dev_results_path, results_dev)

        # validate dev dataset
        manually_validate(dev_results_path, dataset="dev")

        # # load images
        # # run detection
        # image = None
        # results = model.detect([image], verbose=1)

        # # Visualize results
        # # r = results[0]
        # # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


