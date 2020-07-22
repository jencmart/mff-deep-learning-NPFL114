#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net


class SavingCallback(tf.keras.callbacks.Callback):

    def __init__(self, filepath, model, test_dataset, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0
        self.logdir = filepath
        self.model = model
        self.test_dataset = test_dataset

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    pass
                else:
                    if self.monitor_op(current, self.best): # improved
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f, generating result file' % (epoch + 1, self.monitor, self.best,   current))
                        self.best = current

                        # Generate test set annotations, but in args.logdir to allow parallel execution.
                        with open(os.path.join(self.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
                            test_probabilities = self.model.predict(self.test_dataset)
                            for probs in test_probabilities:
                                print(np.argmax(probs), file=out_file)



                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")

    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="dropout rate.")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate.")
    parser.add_argument("--finetune",  default=None, help="Fine-tune last two convolutions...")

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

    # ###############
    # LOAD THE DATA #
    # ###############
    cags = CAGS()
    train_dataset = CAGS.create_dataset(args.batch_size, augment=True)
    dev_dataset = CAGS.create_dataset(args.batch_size, augment=False)
    test_dataset = CAGS.create_dataset(args.batch_size, augment=False)

    # ################################
    # Load the EfficientNet-B0 model #
    # ################################
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False, dynamic_shape=False)
    efficientnet_b0.trainable = False

    if args.finetune == "top":
        for layer in efficientnet_b0.layers[225:]:  # top conv + 7a last conv
            # print(layer.name)
            layer.trainable = True

    if args.finetune == "7a":
        for layer in efficientnet_b0.layers[214:]:
            # print(layer.name)
            layer.trainable = True

    # tf.keras.layers.Layer has a mutable trainable property
    #  after changing it, you need to call .compile again
    # Furhtermore
    # training argument passed to the invocation call decides whether
    # the layer is executed in training regime
    # == neurons gets dropped in dropout, batch normalization computes estimates on the batch)
    # or in inference regime.
    # exception: BatchNorm if trainable==False .... runs in inference even when training==true

    # ############
    # CUSTOM TOP #
    # ############
    x = efficientnet_b0.output[0]  # pooled and flattened ??? MaxPooling nebo AvgPooling ???
    x = tf.keras.layers.Flatten()(x)
    if args.dropout_rate > 0:
        x = tf.keras.layers.Dropout(args.dropout_rate , name="dropout_out")(x)
    predictions = tf.keras.layers.Dense(CAGS.num_classes, activation='softmax', name="fc_out")(x)
    model = tf.keras.Model(inputs=efficientnet_b0.input, outputs=predictions)

    # #########
    # Compile #
    ###########
    optimizer = tf.optimizers.Adam(lr=0.0001)  # tf.optimizers.RMSprop(lr=2e-5)
    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    print(model.summary())

    # #######
    # Train #
    # #######
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    saving_callback = SavingCallback(args.logdir,model,test_dataset)
    model.fit(train_dataset,
              epochs=args.epochs,
              validation_data=dev_dataset,
              callbacks=[tb_callback, saving_callback]
              )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as out_file:
        test_probabilities = model.predict(test_dataset)
        for probs in test_probabilities:
            print(np.argmax(probs), file=out_file)
