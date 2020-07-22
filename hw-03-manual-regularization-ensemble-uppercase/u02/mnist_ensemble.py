#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layers", default="200", type=str, help="Hidden layer configuration.")
    parser.add_argument("--models", default=7, type=int, help="Number of models.")
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

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load data
    mnist = MNIST()

    # Create models
    models = []
    for model_cnt in range(args.models):
        if args.recodex:
            tf.keras.utils.get_custom_objects()["glorot_uniform"] = tf.initializers.GlorotUniform(seed=args.seed + model_cnt)
            tf.keras.utils.get_custom_objects()["orthogonal"] = tf.initializers.Orthogonal(seed=args.seed + model_cnt)
            tf.keras.utils.get_custom_objects()["uniform"] = tf.initializers.RandomUniform(seed=args.seed + model_cnt)

        models.append(tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        ] + [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for hidden_layer in args.hidden_layers] + [
            tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
        ]))

        models[-1].compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        print("Training model {}: ".format(model_cnt + 1), end="", flush=True)
        models[-1].fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs, verbose=0
        )
        print("Done")

    with open("mnist_ensemble.out", "w") as out_file:
        for model_cnt in range(args.models):
            # Compute the accuracy on the dev set for
            # the individual `models[model]`.
            individual_accuracy = (models[model_cnt].evaluate(mnist.dev.data["images"], mnist.dev.data["labels"], batch_size=args.batch_size))[1]

            if model_cnt == 0:
                print("{:.2f} {:.2f}".format(100 * individual_accuracy, 100 * individual_accuracy), file=out_file)
                # print("{}: {:.2f} {:.2f}".format(model_cnt, 100 * individual_accuracy, 100 * individual_accuracy)) # todo - debug print
                continue

            # Compute the accuracy on the dev set for the ensemble `models[0:model+1].

            # special_input_layer = tf.keras.Input(shape=(MNIST.H, MNIST.W, MNIST.C,))
            special_input_layer = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])

            # obtain inputs and outputs of the individual models
            list_of_outputs = []
            for i in range(model_cnt + 1):
                list_of_outputs.append(models[i](special_input_layer))

            # create the average layer
            avg_outputs = tf.keras.layers.average(list_of_outputs)
            ensemble_model = tf.keras.Model(inputs=special_input_layer, outputs=avg_outputs)

            # compile the model
            ensemble_model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

            # evaluate the model
            ensemble_accuracy = (ensemble_model.evaluate(mnist.dev.data["images"], mnist.dev.data["labels"], batch_size=args.batch_size) )[1]

            print("{:.2f} {:.2f}".format(100 * individual_accuracy, 100 * ensemble_accuracy), file=out_file)
            # print("{}: {:.2f} {:.2f}".format(model_cnt, 100 * individual_accuracy, 100 * ensemble_accuracy)) # todo - debug print

