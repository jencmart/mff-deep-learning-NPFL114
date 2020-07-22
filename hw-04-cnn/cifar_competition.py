#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10


def simple_VGG():
    # simplified VGG
    # our images are 7times smaller than  vgg (32 x 32)
    # start with less channels
    model = tf.keras.models.Sequential()

    # 1
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(CIFAR10.H, CIFAR10.W, CIFAR10.C))) # 32,32,3
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))  # 16

    # 2
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))  # 8

    # 3
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))  # 4
    model.add(tf.keras.layers.BatchNormalization())

    # 2x dense
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    # model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
    # model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))

    # last dense
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    opt = tf.optimizers.Adam()  # lr001 momentum09
    model.compile(
        optimizer=opt,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # TODO: Define reasonable defaults and optionally more parameters
    parser.add_argument("--batch_size", default=65, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
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
    args.logdir = "logs"

    # Load data
    cifar = CIFAR10()
    model = simple_VGG()

    model.fit(
        cifar.train.data["images"], cifar.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)



