import os
import sys
import urllib.request
import zipfile

import numpy as np

# Loads the Uppercase data.
# - The data consists of three Datasets
#   - train
#   - dev
#   - test [all in lowercase]
# - When loading, maximum number of alphabet characters can be specified,
#   in which case that many most frequent characters will be used, and all
#   other will be remapped to "<unk>".
# - Batches are generated using a sliding window of given size,
#   i.e., for a character, we includee left `window` characters, the character
#   itself and right `window` characters, `2 * window + 1` in total.
class UppercaseData:
    LABELS = 2

    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/datasets/uppercase_data.zip"

    class Dataset:
        def __init__(self, data, window, alphabet_size, shuffle_batches, seed=42):
            self._window = window
            self._text = data
            self._size = len(self._text)

            # 1. Create alphabet_map ( All characters or only most common)
            alphabet_map = {"<pad>": 0, "<unk>": 1}
            if not isinstance(alphabet_size, int):
                # use whole alphabet ...
                for index, letter in enumerate(alphabet_size):
                    alphabet_map[letter] = index
            else:
                # else find most frequent characters
                freqs = {}
                for char in self._text.lower():
                    freqs[char] = freqs.get(char, 0) + 1

                most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
                for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):

                    alphabet_map[char] = i
                    if alphabet_size and len(alphabet_map) >= alphabet_size:
                        break

            # 2. Remap lower-cased input characters using the alphabet_map
            lcletters = np.zeros(self._size + 2 * window, np.int16)  # delka dat + 2*velikost okenka

            for i in range(self._size):
                # for every letter in the data
                char = self._text[i].lower() # todo -- tady kazdy pismenko lovercasuji

                if char not in alphabet_map:  # map unknown ( characters not in map ) to <unk>
                    char = "<unk>"
                # todo -- a pokladam je "window" mezer od sebe ???
                lcletters[i + window] = alphabet_map[char]  # na pozici i+windows put char from alphabet ( integer )
                                                            # defaultly na pozici i
            # 3. Generate batches data
            windows = np.zeros([self._size, 2 * window + 1], np.int16)  # default je to [size, 1 ]
            labels = np.zeros(self._size, np.uint8)
            for i in range(self._size):
                windows[i] = lcletters[i:i + 2 * window + 1]  # todo -- co je na techto pozicich ???
                labels[i] = self._text[i].isupper()  # prvni v okenku je velke ---> velke
            self._data = {"windows": windows, "labels": labels}

            # 4. Compute alphabet todo -- to vubec nechapu tohle
            self._alphabet = [None] * len(alphabet_map)
            for key, value in alphabet_map.items():
                self._alphabet[value] = key

            self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

        @property
        def alphabet(self):
            return self._alphabet

        @property
        def text(self):
            return self._text

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


    def __init__(self, window, alphabet_size=0):
        path = os.path.basename(self._URL)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(self._URL, filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    data = dataset_file.read().decode("utf-8")
                setattr(self, dataset,

                        self.Dataset(
                            data,
                            window,
                            alphabet_size=alphabet_size if dataset == "train" else self.train.alphabet,
                            shuffle_batches=dataset == "train",
                        ))
