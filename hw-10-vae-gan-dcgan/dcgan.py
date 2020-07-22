#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

# 47b0acaf-eb3e-11e9-9ce9-00505601122b
# 8194b193-e909-11e9-9ce9-00505601122b

# The neural network model
class Network:
    def __init__(self, args):
        self._seed = args.seed
        self._z_dim = args.z_dim

        # TODO: Define `self.generator` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        input = tf.keras.Input(shape=[args.z_dim], name="gen_in")
        x = input

        # - applies batch normalized dense layer with 1024 units and ReLU (do not forget about `use_bias=False` in suitable places)
        x = tf.keras.layers.Dense(units=1024, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        # - applies batch normalized dense layer with MNIST.H // 4 * MNIST.W // 4 * 64 units and ReLU
        x = tf.keras.layers.Dense(units=MNIST.H // 4 * MNIST.W // 4 * 64, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        # - reshapes the current hidder output to [MNIST.H // 4, MNIST.W // 4, 64]
        x = tf.keras.layers.Reshape([MNIST.H // 4, MNIST.W // 4, 64])(x)

        # - applies batch normalized transposed convolution with 32 filters, kernel size 5, stride 2, same padding, and ReLU activation (again `use_bias=False`)
        x = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=5,strides=2,padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        # - applies transposed convolution with 1 filters, kernel size 5, stride 2, same padding, and sigmoid activation
        output = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=5,strides=(2,2),padding="same", activation=tf.nn.sigmoid)(x)

        self.generator = tf.keras.Model([input], [output], name="gan_generator")

        # ##############################################################################################################
        # TODO: Define `self.discriminator` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        input = tf.keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C], name="disc_in")
        x = input

        # - computes batch normalized convolution with 32 filters, kernel size 5, same padding, and ReLU activation (do not forget `use_bias=False` where appropriate)
        x = tf.keras.layers.Conv2D(filters=32,kernel_size=5, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        # - max-pools with kernel size 2 and stride 2
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

        # - computes batch normalized convolution with 64 filters, kernel size 5, same padding, and ReLU activation (again `use_bias=False`)
        x = tf.keras.layers.Conv2D(filters=64,kernel_size=5, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)


        # - max-pools with kernel size 2 and stride 2
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

        # - flattens the current representation
        x = tf.keras.layers.Flatten()(x)
        # - applies batch normalized dense layer with 1024 units and ReLU activation (`use_bias`)
        x = tf.keras.layers.Dense(units=1024, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        # - applies output dense layer with one output and a suitable activation function
        output = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(x)
        self.discriminator = tf.keras.Model([input], [output], name="gan_discriminator")

        self._generator_optimizer, self._discriminator_optimizer = tf.optimizers.Adam(), tf.optimizers.Adam()
        self._loss_fn = tf.losses.BinaryCrossentropy()
        self._discriminator_accuracy = tf.metrics.Mean()
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _sample_z(self, batch_size):
        """Sample random latent variable."""
        return tf.random.uniform([batch_size, self._z_dim], -1, 1, seed=self._seed)

    @tf.function
    def train_batch(self, images):

        # TODO: Generator training. Using a Gradient tape:
        with tf.GradientTape() as tape:
            # - generate random images using a `generator`; do not forget about `training=True`
            # - run discriminator on the generated images, also using `training=True`
            # (even if not updating discriminator parameters, we want to perform possible BatchNorm in it)
            z = self._sample_z(args.batch_size)
            genrated_images_from_random_Z = self.generator(z, training=True)
            discriminator_output = self.discriminator(genrated_images_from_random_Z,  training=True)

            # - compute loss using `_loss_fn`, with target labels `tf.ones_like(discriminator_output)`
            generator_loss = self._loss_fn(y_true=tf.ones_like(discriminator_output), y_pred=discriminator_output)

        # Then, compute the gradients with respect to generator trainable variables and update
        # generator trainable weights using self._generator_optimizer.
        variables = self.generator.trainable_variables
        model_gradients = tape.gradient(generator_loss, variables)
        self._generator_optimizer.apply_gradients(zip(model_gradients, variables))

        # TODO: Discriminator training. Using a Gradient tape:
        with tf.GradientTape() as tape:
            # - discriminated_real = discriminate `images` with `training=True`
            # - discriminated_fake = discriminate images generated in generator training with `training=True`
            discriminated_real = self.discriminator(images, training=True)
            discriminated_fake = self.discriminator(genrated_images_from_random_Z, training=True)

            loss_real = self._loss_fn(y_true=tf.ones_like(discriminated_real), y_pred=discriminated_real)
            loss_fake = self._loss_fn(y_true=tf.zeros_like(discriminated_fake), y_pred=discriminated_fake)
            discriminator_loss = tf.reduce_sum(loss_real) + tf.reduce_sum(loss_fake)

        # Then, compute the gradients with respect to discriminator trainable variables and update
        # discriminator trainable weights using self._discriminator_optimizer.
        variables = self.discriminator.trainable_variables
        model_gradients = tape.gradient(discriminator_loss, variables)
        self._discriminator_optimizer.apply_gradients(zip(model_gradients, variables))

        if self._discriminator_optimizer.iterations % 100 == 0:
            tf.summary.experimental.set_step(self._discriminator_optimizer.iterations)
            self._discriminator_accuracy.reset_states()
            self._discriminator_accuracy(tf.greater(discriminated_real, 0.5))
            self._discriminator_accuracy(tf.less(discriminated_fake, 0.5))
            with self._writer.as_default():
                tf.summary.scalar("dcgan/generator_loss", generator_loss)
                tf.summary.scalar("dcgan/discriminator_loss", discriminator_loss)
                tf.summary.scalar("dcgan/discriminator_accuracy", self._discriminator_accuracy.result())

        return generator_loss + discriminator_loss

    def generate(self):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.generator(self._sample_z(GRID * GRID))

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = tf.stack([-1 * tf.ones(GRID), tf.linspace(-1., 1., GRID)], -1)
            ends = tf.stack([1 * tf.ones(GRID), tf.linspace(-1., 1., GRID)], -1)
        else:
            # Generate random Z
            starts, ends = self._sample_z(GRID), self._sample_z(GRID)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self.generator(interpolated_z)

        # Stack the random images, then an empty row, and finally interpolated imates
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self._writer.as_default():
            tf.summary.image("dcgan/images", tf.expand_dims(image, 0))

    def train_epoch(self, dataset, args):
        loss = 0
        for batch in dataset.batches(args.batch_size):
            loss += self.train_batch(batch["images"])
        self.generate()
        return loss


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
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
    mnist = MNIST(args.dataset)

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        loss = network.train_epoch(mnist.train, args)

    with open("gan.out", "w") as out_file:
        print("{:.2f}".format(loss), file=out_file)
