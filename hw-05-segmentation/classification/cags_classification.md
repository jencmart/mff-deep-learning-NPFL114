use pretrained EfficientNet-B0 model to achieve best accuracy in CAGS classification.

The [CAGS dataset]consists of images of cats/dogs of size $224×224$, each classified in one of 34 breeds + mask 
To load the dataset, use the [cags_dataset.py] module.
 The dataset is stored in a [TFRecord file](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
and each element is encoded as a [tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example).
Therefore the dataset is loaded using `tf.data` API
each entry can be decoded using `.map(CAGS.parse)` call.

To load the EfficientNet-B0, use the the provided
[efficient_net.py](https://github.com/ufal/npfl114/tree/master/labs/05/efficient_net.py)
module. Its method `pretrained_efficientnet_b0(include_top)`:
- downloads the pretrained weights if they are not found;
- it returns a `tf.keras.Model` processing image of shape $(224, 224, 3)$ with
  float values in range $[0, 1]$ and producing a list of results:
  - the first value is the final network output:
   
  - the rest of outputs are the intermediate results of the network just before
    a convolution with $\textit{stride} > 1$ is performed (denoted $C_5,
    C_4, C_3, C_2, C_1$ in the Object Detection lecture).

An example performing classification of given images is available in
[image_classification.py](https://github.com/ufal/npfl114/tree/master/labs/05/image_classification.py).

finetuning:
each `tf.keras.layers.Layer` has a mutable `trainable` property indicating whether its variables should be updated 
– however, after changing it, you need to call `.compile` again 
(or otherwise make sure the list of trainable variables for the optimizer is updated). 

Furthermore, `training` argument passed to the invocation call decides 
whether the layer is executed in training regime 
(neurons gets dropped in dropout, batch normalization computes estimates on the batch) or in inference regime.
There is one exception though – if `trainable == False` on a batch normalization layer, it runs in the
inference regime even when `training == True`._
