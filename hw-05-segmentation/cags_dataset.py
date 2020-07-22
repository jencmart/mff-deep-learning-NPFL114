import os
import sys
import urllib.request

import tensorflow as tf
import tensorflow_addons as tfa


class CAGS:
    H, W, C = 224, 224, 3
    num_classes = 34
    num_classes_types = 1  # just a single type right now
    LABELS = [
        # Cats
        # 0            1         2            3                 4
        "Abyssinian", "Bengal", "Bombay", "British_Shorthair", "Egyptian_Mau",
        #   5              6              7           8
        "Maine_Coon", "Russian_Blue", "Siamese", "Sphynx",
        # Dogs
        "american_bulldog", "american_pit_bull_terrier", "basset_hound",
        "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
        "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
        "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
        "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
        "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
        "wheaten_terrier", "yorkshire_terrier",
    ]

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/"

    @staticmethod
    def preprocess(x, y):
        # serialized_example["image"], [serialized_example["mask"], serialized_example["label"]]
        image = x  # serialized_example["image"]
        # label = serialized_example["label"]
        mask = y  # [0]  # serialized_example["mask"]

        # image+ mask   ................
        if tf.random.uniform([]) >= 0.5:  # random horizontal flip
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if tf.random.uniform([]) >= 0.5:  # random centrl crop
            # fraction = tf.random.uniform([],minval=0.7, maxval=0.9) # leave this amount of the image ...
            image = tf.image.central_crop(image, central_fraction=0.8)
            mask = tf.image.central_crop(mask, central_fraction=0.8)

            image = tf.image.resize(image, (224, 224))
            mask = tf.image.resize(mask, (224, 224))

        # image       .................
        if tf.random.uniform([]) >= 0.5:  # random ajust saturation
            factor = tf.random.uniform([], minval=0.3, maxval=3.)  # Factor to multiply the saturation by ( 0 == BW)
            image = tf.image.adjust_saturation(image, factor)

        if tf.random.uniform([]) >= 0.5:  # random ajust saturation
            add = tf.random.uniform([], minval=0.1, maxval=0.3)  # A scalar. Amount to add to the pixel values.
            image = tf.image.adjust_brightness(image, add)

        # serialized_example["image"] = image
        # serialized_example["mask"] = mask

        return x, y

    @staticmethod
    def parse(
            serialized_example):  # be fucking aware ... feature names are stored in dataset !!! you can't have arbitrary names here
        keys_to_features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)}

        serialized_example = tf.io.parse_single_example(serialized_example, keys_to_features)
        serialized_example["image"] = tf.image.convert_image_dtype(
            tf.image.decode_jpeg(serialized_example["image"], channels=3), tf.float32)
        serialized_example["mask"] = tf.image.convert_image_dtype(
            tf.image.decode_png(serialized_example["mask"], channels=1), tf.float32)
        # serialized_example["label"] = tf.cast(serialized_example["label"], tf.int32)
        if serialized_example["label"] <= 8:
            serialized_example["label"] = 0
        else:
            serialized_example["label"] = 1
        # return example
        return serialized_example["image"], serialized_example["mask"]

    @staticmethod
    def create_dataset(dataset, batch_size, shuffle=False, augment=False, repeat_augmented=6):
        dataset = dataset.map(CAGS.parse)

        print("creating the dataset")
        if augment:
            if repeat_augmented is not None:
                # print("repeating the dataset {} times".format(repeat_augmented))
                dataset = dataset.repeat(repeat_augmented)  # repeat
            dataset = dataset.shuffle(
                10000 + 3 * batch_size)  # 10000 + 3 * batch_size [43 batches po 50 images ... 450 training datas ??]
            dataset = dataset.map(CAGS.preprocess)  # augment

        dataset = dataset.batch(batch_size)

        return dataset

    def __init__(self):
        base_path = "cags"
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        for dataset, size in [("train", 57463494), ("dev", 8138328), ("test", None), ("both", None)]:
            file_name = "cags.{}.tfrecord".format(dataset)
            path = os.path.join(base_path, file_name)
            if dataset == "both":
                pass
            elif not os.path.exists(path) or (size is not None and os.path.getsize(path) != size):
                print("Downloading file {}...".format(file_name), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, file_name), filename=path)

            if dataset == "both":
                path = []
                file_name = "cags.{}.tfrecord".format("train")
                path.append(os.path.join(base_path, file_name))

                file_name = "cags.{}.tfrecord".format("dev")
                path.append(os.path.join(base_path, file_name))
                # print("setting both")
                # print(path)
                setattr(self, dataset, tf.data.TFRecordDataset(path))
            else:
                setattr(self, dataset, tf.data.TFRecordDataset(path))
