#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST


import datetime

class MyCustomCallback(tf.keras.callbacks.Callback):

  # def on_train_batch_begin(self, batch, logs=None):
  #   print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    # print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    print("learning rate: {:.5f}".format(self.model.optimizer.learning_rate(model.optimizer.iterations)))

  # def on_test_batch_begin(self, batch, logs=None):
  #   print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  #
  # def on_test_batch_end(self, batch, logs=None):
  #   print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")

    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")

    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")

    parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")

    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
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

    # Load data
    mnist = MNIST()

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])

    #  If `args.decay`
    if args.decay is None:
        used_learning_rate = args.learning_rate

    else:
        learning_rate = args.learning_rate
        # `decay_steps` should be total number of training batches
        #  training MNIST dataset size =  `mnist.train.size` , assume divisibility by `args.batch_size`.

        learning_rate_final = args.learning_rate_final

        if args.decay == 'polynomial':
            decay_steps = mnist.train.size * args.epochs / args.batch_size
            used_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
                learning_rate,
                decay_steps,
                end_learning_rate=learning_rate_final,
                # power=1.0,
                # cycle=False
            )

        else:  # exponential
            # todo  set `decay_rate` appropriately to reach `args.learning_rate_final`
            decay_steps = mnist.train.size / args.batch_size
            decay_rate = (learning_rate_final / learning_rate) ** (1/args.epochs)
            used_learning_rate = tf.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps,
                decay_rate=decay_rate,
                staircase=False  # (and keep the default `staircase=False`).
            )

    if args.optimizer == "SGD":     # For `SGD`, `args.momentum` can be specified.
        momentum = 0.0
        if args.momentum is not None:
            momentum = args.momentum
        opt = tf.keras.optimizers.SGD(learning_rate=used_learning_rate, momentum=momentum)

    else:  # Adam
        if args.momentum is not None:
            momentum = args.momentum
        opt = tf.keras.optimizers.Adam(learning_rate=used_learning_rate)

    model.compile(
        optimizer=opt,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)
    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback]  # MyCustomCallback()
    )

    test_logs = model.evaluate(
        mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
    )
    tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(model.metrics_names, test_logs)})

    accuracy = test_logs[1]  # Write test accuracy as percentages rounded to two decimal places.

    with open("mnist_training.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)

    #   If a learning rate schedule is used, you can find out the current learning
    #   rate by using `model.optimizer.learning_rate(model.optimizer.iterations)`,
    #   so after training this value should be `args.learning_rate_final`.

