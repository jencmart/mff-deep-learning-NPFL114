#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf

from mnist import MNIST


class Model(tf.Module):

    def __init__(self, args):
        self._W1 = tf.Variable(tf.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed), trainable=True)
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)

        self._W2 = tf.Variable(tf.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed))
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]))

    def predict(self, inputs):
        input_layer = tf.reshape(inputs, [inputs.shape[0], -1])  # shape 0 == batch pieces
        hidden_layer = tf.nn.tanh(tf.math.add(tf.linalg.matmul(input_layer, self._W1), self._b1))
        output_layer = tf.nn.softmax(tf.math.add(tf.linalg.matmul(hidden_layer, self._W2), self._b2))
        return output_layer

    def train_epoch(self, dataset):

        for batch in dataset.batches(args.batch_size):

            batch_x = tf.reshape(batch["images"], [batch["images"].shape[0], -1])
            batch_y = tf.one_hot(batch["labels"], MNIST.LABELS)  # 50 x 10

            gradient_by_W1, gradient_by_W2, gradient_by_b1, gradient_by_b2 = self.run(batch_x, batch_y)

            # MANUAL update
            self._W1 = tf.compat.v1.assign_sub(self._W1, tf.math.scalar_mul(args.learning_rate, gradient_by_W1))
            self._W2 = tf.compat.v1.assign_sub(self._W2, tf.math.scalar_mul(args.learning_rate, gradient_by_W2))
            self._b1 = tf.compat.v1.assign_sub(self._b1, tf.math.scalar_mul(args.learning_rate, gradient_by_b1))
            self._b2 = tf.compat.v1.assign_sub(self._b2, tf.math.scalar_mul(args.learning_rate, gradient_by_b2))


    def evaluate(self, dataset):
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(args.batch_size):
            # Compute the probabilities of the batch images
            output_layer = self.predict(batch["images"])
            probabilities = output_layer
            # Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            predicted_labels = tf.math.argmax(probabilities, axis=1)
            true_labels = batch["labels"]
            res = predicted_labels - true_labels
            correct += res.shape[0] - tf.math.count_nonzero(res)
        return correct / dataset.size

    def predict_for_train(self, input_vector):
        # dense + tanh
        hidden_layer_before_tanh = tf.matmul(input_vector, self._W1)
        hidden_layer_before_tanh = tf.add(hidden_layer_before_tanh, self._b1)
        hidden_layer = tf.nn.tanh(hidden_layer_before_tanh)

        # dense + softmax
        output_layer = tf.matmul(hidden_layer, self._W2)
        output_layer = tf.add(output_layer, self._b2)
        probabilities = tf.math.softmax(output_layer)

        return probabilities, hidden_layer_before_tanh, hidden_layer

    def run(self, batch_x, batch_y):
        probabilities, hidden_layer_before_tanh, hidden_layer = self.predict_for_train(batch_x)

        # ############## BY  B2 ###############################
        xent_softmax = probabilities - batch_y
        manual_gradient_by_b2 = tf.math.reduce_mean(xent_softmax, axis=0)

        # ############## BY  W2 ###############################
        manual_gradient_by_W2 = tf.einsum("ai,aj->aij", hidden_layer, xent_softmax)
        manual_gradient_by_W2 = tf.math.reduce_mean(manual_gradient_by_W2, axis=0)

        # ############## BY  B1 ###############################
        xent_softmax_w2 = xent_softmax @ tf.transpose(self._W2)
        tanh = tf.math.reciprocal(tf.math.square(tf.cosh(hidden_layer_before_tanh)))
        xent_softmax_w2_tanh = tf.multiply(xent_softmax_w2, tanh)
        manual_gradient_by_b1 = tf.math.reduce_mean(xent_softmax_w2_tanh, axis=0)

        # ############## BY  W1 ###############################
        #   `C[a, i, j] = A[a, i] * B[a, j]` USE   `tf.einsum("ai,aj->aij", A, B)`
        manual_gradient_by_W1 = tf.einsum("ai,aj->aij", batch_x, xent_softmax_w2_tanh)
        manual_gradient_by_W1 = tf.math.reduce_mean(manual_gradient_by_W1, axis=0)

        return manual_gradient_by_W1, manual_gradient_by_W2, manual_gradient_by_b1, manual_gradient_by_b2


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--learning_rate", default=0.2, type=float, help="Learning rate.")
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
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)

    # Create the model
    model = Model(args)

    # run in 10 times !!! oh my god !!!
    for epoch in range(args.epochs):
        # Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(mnist.train)

        # (sgd_backpropagation): Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = model.evaluate(mnist.dev)

        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default():
            tf.summary.scalar("dev/accuracy", 100 * accuracy, step=epoch + 1)

    # (sgd_backpropagation): Evaluate the test data using `evaluate` on `mnist.test` dataset
    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default():
        tf.summary.scalar("test/accuracy", 100 * accuracy, step=epoch + 1)

    # Save the test accuracy in percents rounded to two decimal places.
    with open("sgd_manual.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)