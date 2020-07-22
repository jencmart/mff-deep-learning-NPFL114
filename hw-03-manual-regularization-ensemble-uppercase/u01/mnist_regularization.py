#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
    parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
    parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

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

    # Load data
    mnist = MNIST()

    # Implement L2 regularization.
    # If `args.l2` is nonzero,
    l2_regularizer = None
    if args.l2 != 0:
        # create a `tf.keras.regularizers.L1L2` regularizer
        l2_regularizer = tf.keras.regularizers.L1L2(l2=args.l2)
        # use it for all kernels ( NOT FOR BIASES ) of all Dense layers  (last one included)

    # Implement dropout.
    # Add a `tf.keras.layers.Dropout`
    dropout_layer = None
    if args.dropout > 0:
        dropout_layer = args.dropout
        # with `args.dropout` rate after the Flatten
        # layer and after each Dense hidden layer (but not after the output Dense layer).

    # Create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]))
    if dropout_layer is not None:
        model.add(tf.keras.layers.Dropout(dropout_layer))
    for hidden_layer in args.hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu, kernel_regularizer=l2_regularizer))
        if dropout_layer is not None:
            model.add(tf.keras.layers.Dropout(dropout_layer))
    model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax, kernel_regularizer=l2_regularizer))

    # Implement label smoothing.

    loss = tf.losses.SparseCategoricalCrossentropy()
    accuracy = tf.metrics.SparseCategoricalAccuracy(name="accuracy")

    if args.label_smoothing > 0:
        #  change the`SparseCategorical{Crossentropy,Accuracy}` to `Categorical{Crossentropy,Accuracy}`
        # because `label_smooting` is supported only by `CategoricalCrossentropy`.
        loss = tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        accuracy = tf.metrics.CategoricalAccuracy(name="accuracy")

        # you also need to modify the labels of all three datasets to full categorical distribution
        # (i.e., `mnist.{train,dev,test}.data["labels"]`) from indices of the gold class
        # you can use either NumPy or there is a helper method in `tf.keras.utils`
        mnist.train.data["labels"] = tf.keras.utils.to_categorical( mnist.train.data["labels"], num_classes=None, dtype='float32')
        mnist.dev.data["labels"] = tf.keras.utils.to_categorical( mnist.dev.data["labels"], num_classes=None, dtype='float32')
        mnist.test.data["labels"] = tf.keras.utils.to_categorical( mnist.test.data["labels"], num_classes=None, dtype='float32')

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=loss,
        metrics=[accuracy],
    )

    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    model.fit(
        mnist.train.data["images"][:5000], mnist.train.data["labels"][:5000],  # todo -- labels train
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),  # todo -- labels dev
        callbacks=[tb_callback],
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,  # todo -- labels test
    )
    tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(model.metrics_names, test_logs)})

    accuracy = test_logs[model.metrics_names.index("accuracy")]
    with open("mnist_regularization.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)