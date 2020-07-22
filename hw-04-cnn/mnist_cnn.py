#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:

    def create_layer(self, input_layer, layer_spec):
        if layer_spec[:2] == "C-":  # Add a conv layer with ReLU
            layer_data = layer_spec.split("-")
            filters = int(layer_data[1])  # 10
            kernel_size = int(layer_data[2])  # 3
            stride = int(layer_data[3])  # 1
            padding = layer_data[4]  # same
            output = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(stride, stride),
                                            padding=padding, activation=tf.keras.activations.relu)(input_layer)

        elif layer_spec[:2] == "CB":  # Same, but use batch normalization.
            layer_data = layer_spec.split("-")
            filters = int(layer_data[1])  # 10
            kernel_size = int(layer_data[2])  # 3
            stride = int(layer_data[3])  # 1
            padding = layer_data[4]  # same
            convoluted = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(stride, stride),
                                            padding=padding, activation=None,use_bias=False)(input_layer)
            normed = tf.keras.layers.BatchNormalization()(convoluted)
            output = tf.keras.activations.relu(normed)

        elif layer_spec[0] == "M":  # Add max pooling with specified size and stride
            layer_data = layer_spec.split("-")
            size = int(layer_data[1])  # 3
            stride = int(layer_data[2])  # 2
            output = tf.keras.layers.MaxPool2D(pool_size=(size, size), strides=stride)(input_layer)

        elif layer_spec[0] == "F":  # Flatten inputs. Must appear exactly once in the architecture.
            output = tf.keras.layers.Flatten()(input_layer)

        elif layer_spec[0] == "H":
            layer_data = layer_spec.split("-")
            units = int(layer_data[1])  # 100
            output = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu)(input_layer)

        else:  # Droput layer  Example: `D-0.5`
            layer_data = layer_spec.split("-")
            rate = float(layer_data[1])
            output = tf.keras.layers.Dropout(rate=rate)(input_layer)
        return output

    def __init__(self, args):
        # TODO: Create the model. The template uses functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        hidden = inputs

        tmp_list = re.split('R|]', args.cnn) # C-16-3-2-same,R-[C-16-3-1-same,C-16-3-1-same],M-3-2,F,H-100
        list_of_layers = []

        for x in tmp_list:
            if x[0] != '-':
                res = x.split(",")
                for tmp in res:
                    if tmp != "":
                        list_of_layers.append(tmp)
            else:
                list_of_layers.append(str(x[1:]))

        for layer_spec in list_of_layers:
            if layer_spec == "":
                continue

            # print(layer_spec)
            if layer_spec[0] == "[":  # Add a residual connection R-[C-16-3-1-same,C-16-3-1-same]
                list_of_layers_for_residual = layer_spec[1:].split(",")

                hidden_shortcut = hidden
                for R_layer_spec in list_of_layers_for_residual:
                    hidden = self.create_layer(hidden, R_layer_spec)
                hidden = tf.keras.layers.Add()([hidden, hidden_shortcut])

            else:
                hidden = self.create_layer(hidden, layer_spec)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self._model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        self._tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    def train(self, mnist, args):
        self._model.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self._tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self._model.evaluate(mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self._tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(self._model.metrics_names, test_logs)})
        return test_logs[self._model.metrics_names.index("accuracy")]


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default="C-16-3-2-same,R-[C-16-3-1-same,C-16-3-1-same],M-3-2,F,H-100", type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")  # 30
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed)
        tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed)
        tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed)

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
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
