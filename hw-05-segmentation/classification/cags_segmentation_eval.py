#!/usr/bin/env python3
import argparse
import os

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS



# class MyShittyIoUMetric(tf.keras.metrics.Metric):
#
#     def __init__(self, name='categorical_true_positives', **kwargs):
#       super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
#       self.true_positives = self.add_weight(name='tp', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#       y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
#       values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
#       values = tf.cast(values, 'float32')
#       if sample_weight is not None:
#         sample_weight = tf.cast(sample_weight, 'float32')
#         values = tf.multiply(values, sample_weight)
#       self.true_positives.assign_add(tf.reduce_sum(values))
#
#     def result(self):
#       return self.true_positives
#
#     def reset_states(self):
#       # The state of the metric will be reset at the start of each epoch.
#       self.true_positives.assign(0.)




class CAGSMaskIoU(tf.metrics.Mean):
    """CAGSMaskIoU computes IoU for CAGS dataset masks predicted by binary classification"""

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, CAGS.H * CAGS.W])
        y_pred_mask = tf.reshape(tf.math.round(y_pred) == 1, [-1, CAGS.H * CAGS.W])

        intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
        union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

        intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
        union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

        iou = tf.where(union == 0, 1., intersection / union)
        return super().update_state(iou, sample_weight)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str, help="Path to predicted output.")
    parser.add_argument("dataset", type=str, help="Which dataset to evaluate ('dev', 'test').")
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

    # Load the gold data
    gold_masks = getattr(CAGS(), args.dataset).map(CAGS.parse).map(lambda example: example["mask"])

    # Create the metric
    iou = CAGSMaskIoU()

    # Read the predictions
    with open(args.predictions, "r", encoding="utf-8-sig") as predictions_file:
        for gold_mask in gold_masks:
            predicted_runs = [int(run) for run in predictions_file.readline().split()]
            assert sum(predicted_runs) == CAGS.H * CAGS.W

            predicted_mask = np.zeros([CAGS.H * CAGS.W], np.int32)
            offset = 0
            for i, run in enumerate(predicted_runs):
                predicted_mask[offset:offset + run] = i % 2
                offset += run

            iou(gold_mask, predicted_mask)

    print("{:.2f}".format(100 * iou.result()))
