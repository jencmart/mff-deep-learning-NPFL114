### Assignment: svhn_competition
#### Date: Deadline: Apr 26, 23:59
#### Points: 5 points+5 bonus

The goal of this assignment is to implement a system performing object
recognition, optionally utilizing pretrained EfficientNet-B0 backbone.

The [Street View House Numbers (SVHN) dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/demos/svhn_train.html)
annotates for every photo all digits appearing on it, including their bounding
boxes. The dataset can be loaded using the [svhn_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_dataset.py)
module. Similarly to the `CAGS` dataset, it is stored in a
[TFRecord file](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
with [tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
elements, which can be decoded using `.map(SVHN.parse)` call. Every element
is a dictionary with the following keys:
- `"image"`: a square 3-channel image,
- `"classes"`: a 1D tensor with all digit labels appearing in the image,
- `"bboxes"`: a `[num_digits, 4]` 2D tensor with bounding boxes of every digit in the image.

Given that the dataset elements are each of possibly different size, it is
quite tricky to use the `tf.data` API – converting the dataset to NumPy may
make your life easier. 
_Also note that only `tf.` calls should be used the argument of `tf.data.Dataset.map`, 
so if you want to use `bboxes_training` directly
with `tf.data.Dataset.map`, you need to use
[tf.numpy_function](https://www.tensorflow.org/api_docs/python/tf/numpy_function)._


Each test set image annotation consists of a sequence of space separated
five-tuples _label top left bottom right_, and the annotation is considered
correct, if exactly the gold digits are predicted, each with IoU at least 0.5.
The whole test set score is then the prediction accuracy of individual images.
An evaluation of a file with the predictions can be performed by the
[svhn_eval.py](https://github.com/ufal/npfl114/tree/master/labs/06/svhn_eval.py)
module.

usually need at least _35%_ development set accuracy to achieve the required test set performance.

_A baseline solution can use RetinaNet-like single stage detector,
using only a single level of convolutional features (no FPN)
with single-scale and single-aspect anchors. Focal loss is available
as [tfa.losses.SigmoidFocalCrossEntropy](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy)
(using `reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE` option is a good
idea) and non-maximum suppression as
[tf.image.non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression) or
[tf.image.combined_non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression)._
