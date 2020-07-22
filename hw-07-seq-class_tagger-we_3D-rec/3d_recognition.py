#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

import sys
import urllib.request

class ModelNet:
    # The D, H, W are set in the constructor depending
    # on requested resolution and are only instance variables.
    D, H, W, C = None, None, None, 1
    LABELS = [
        "bathtub", "bed", "chair", "desk", "dresser",
        "monitor", "night_stand", "sofa", "table", "toilet",
    ]

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/modelnet{}.npz"

    class Dataset:
        def __init__(self, data, shuffle_batches, seed=42):
            self._data = data
            self._size = len(self._data["voxels"])

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def data(self):
            return self._data

        @property
        def size(self):
            return self._size

        def batches(self, size=None):
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch

    # The resolution parameter can be either 20 or 32.
    def __init__(self, resolution):
        assert resolution in [20, 32], "Only 20 or 32 resolution is supported"

        self.D = self.H = self.W = resolution
        url = self._URL.format(resolution)

        path = os.path.basename(url)
        if not os.path.exists(path):
            print("Downloading {} dataset...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(url, filename=path)

        mnist = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = dict((key[len(dataset) + 1:], mnist[key]) for key in mnist if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train"))



def simple_3D_Net():
    # simplified VGG
    # our images are 7times smaller than  vgg (32 x 32)
    # start with less channels
    model = tf.keras.models.Sequential()


    model.add(tf.keras.layers.Conv3D(48,  kernel_size=6, strides=2, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(ModelNet.D, ModelNet.H, ModelNet.W, 1)))
    model.add(tf.keras.layers.Conv3D(160, kernel_size=5, strides=2, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv3D(512, kernel_size=4, strides=1, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.GlobalMaxPooling3D())
    model.add(tf.keras.layers.Dense(1200, activation='relu'))
    model.add(tf.keras.layers.Dense(len(ModelNet.LABELS), activation='softmax'))

    opt = tf.optimizers.Adam(lr=args.lr)  # lr001 momentum09
    model.compile(
        optimizer=opt,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    #  --batch_size=50 --lr=0.003  --epochs=10
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--lr", default=0.003, type=float)

    parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
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



    # Load the data
    model_net = ModelNet(args.modelnet)
    model = simple_3D_Net()

    # print(model_net.dev.data["labels"])
    # exit(1)
    model.fit(
        model_net.train.data["voxels"], model_net.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(model_net.dev.data["voxels"], model_net.dev.data["labels"]),
    )

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open("3d_recognition.txt", "w", encoding="utf-8") as out_file:
        # TODO: Predict the probabilities on the test set

        for probs in model.predict(model_net.test.data["voxels"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
