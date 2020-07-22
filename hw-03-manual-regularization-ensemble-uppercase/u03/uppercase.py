#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from u03.data.uppercase_data import UppercaseData

def crate_the_model(args):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32))
    model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
    model.add(tf.keras.layers.Flatten())
    if args.dropout is not None:
        model.add(tf.keras.layers.Dropout(args.dropout))
    for hidden_layer in args.hidden_layers:
        if args.dropout is not None:
            model.add(tf.keras.layers.Dropout(args.dropout))
        l2_regularizer = None
        if args.l2 is not None:
            l2_regularizer = tf.keras.regularizers.L1L2(l2=args.l2)

        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu, kernel_regularizer=l2_regularizer))

    l2_regularizer = None
    if args.l2 is not None:
        l2_regularizer = tf.keras.regularizers.L1L2(l2=args.l2)
    model.add(tf.keras.layers.Dense(UppercaseData.LABELS, activation=tf.nn.softmax, kernel_regularizer=l2_regularizer))
    model.summary()

    return model


if __name__ == "__main__":
    # Parse arguments
    # TODO: Set reasonable values for `alphabet_size` and `window`.
    parser = argparse.ArgumentParser()
    # sub 70 NO
    # / 80 / 90 / 100 ... seems to be best options
    # over 130 garbage

    parser.add_argument("--alphabet_size", default=90, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")  # TODO

    # 10  -- 96.20
    # 40  -- 96.26
    # 100 -- 96.37 -----
    # 200 -- 96.30
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")   # todo -- 100 is good
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")  # todo -- 10 is good

    #                                 >>|<<
    # --- batches == 200 -------------------------
    #                          test    val    train
    # 150,150               -- 96.30
    # 300,150               -- 96.02  97.95   98.43    ---- lepsi
    # 500, 150  L2=0.01     -- 100.0  94.92   94.93
    # 500, 150  L2=0.0001   -- 96.77  97.30   97.31
    # 500, 150  L2=0.00001  -- 96.17  97.74   97.80
    # 500, 150              -- 95.99  97.94   98.58    ---- lepsi
    # 300, 200              -- 96.07  97.93   98.44    ---- lepsi
    # hlubsi a tlustsi se zda, z1e nema cenu ....... ( stejne nebo horsi ... )
    parser.add_argument("--hidden_layers", default="150,150,", type=str, help="Hidden layer configuration.")  # todo

    # 2 ---        97.57  97.73
    # 3 ---        97.83  98.25
    # 4 --- 96.05  97.90  98.44  ----
    # 5 --- 96.00  97.93  98.64  ----
    # 6 --- 95.94  97.92  98.64
    parser.add_argument("--window", default=4, type=int, help="Window size to use.")  # TODO TODO TODO

    # learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,

    parser.add_argument("--l2", default=None, type=float, help=" value for l2 regularization")  # todo
    parser.add_argument("--dropout", default=None, type=float, help="value for dropout rate")  # todo

    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=True, action="store_true", help="Verbose TF logging.")
    parser.add_argument("--train", default=True, help="Train or not ?")

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

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # For saving the model
    args.save_dir = "logs"
    args.base_save_name = "weights.best-2.hdf5"  # "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

    # Load data
    uppercase_data = UppercaseData(window=args.window, alphabet_size=args.alphabet_size)

    # Construct the model
    model = crate_the_model(args=args)

    # if train -> train the model
    if args.train:
        print("Compiling the model")

        accuracy = tf.metrics.SparseCategoricalAccuracy(name="accuracy")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            #metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            metrics=[accuracy]
        )

        print("Starting training for {} epochs".format(args.epochs))

        # TensorBoard callback
        tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

        # Saving the model callback (weights only)
        filepath = os.path.join(args.save_dir, args.base_save_name)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        # Train the model
        hist = model.fit(
            uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
            callbacks=[tb_callback, checkpoint_callback],
        )

        # Finally, evaluate the model
        scores = model.evaluate(uppercase_data.test.data["windows"], uppercase_data.test.data["labels"], # batch_size=args.batch_size,
                              )
        tb_callback.on_epoch_end(1, {"val_test_" + metric: value for metric, value in zip(model.metrics_names, scores)})
        # basic print
        print("TEST accuracy is %.2f%%" % (scores[1] * 100))

        # for key in hist.history:
        #     print(key)

    # not train -> load weights
    else:
        # load weights
        filepath = os.path.join(args.save_dir, args.base_save_name)
        model.load_weights(filepath)

        # Compile model (required to make predictions)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        print("Created model and loaded weights from file")

    # Make the final prediction
    X = uppercase_data.test.data["windows"]
    ynew = model.predict_classes(x=X, batch_size=args.batch_size)

    # Load original text
    text = uppercase_data.test.text

    with open(os.path.join(args.save_dir, "uppercase_test.txt"), "w", encoding="utf-8") as out_file:
        for i in range(len(X)):
            # find the correct symbol (just for debug)
            # win = X[i]
            # encoded_char = win[int(len(win) / 2)]
            # char = uppercase_data.test._alphabet[encoded_char]
            char_from_text = text[i]

            if ynew[i] == 0:
               # print("%s" % (char_from_text), end="")
                print("%s" % (char_from_text), end="", file=out_file)

            else:
               # print("%s" % (char_from_text.upper()), end="")
                print("%s" % (char_from_text.upper()), end="", file=out_file)

