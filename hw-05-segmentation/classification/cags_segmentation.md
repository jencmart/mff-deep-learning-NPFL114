
use pretrained EfficientNet-B0 model to
achieve best image segmentation IoU score on the CAGS dataset.

This is an _open-data task_, where you submit only the test set masks
together with the training script 

A mask is evaluated using _intersection over union_ (IoU) metric, which is the
intersection of the gold and predicted mask divided by their union, and the
whole test set score is the average of its masks' IoU. A TensorFlow compatible
metric is implemented by the class `CAGSMaskIoU` of the
[cags_segmentation_eval.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_segmentation_eval.py)
module, which can further be used to evaluate a file with predicted masks.



demo file
generates the test set annotation in the required format –
each mask should be encoded on a single line as a 
space separated sequence of integers 
indicating the length of alternating runs of zeros and ones.
== 7 3 4 ... 7 zeros 3 ones 4 zeros ...

ja delam realnou masku ... posledni vrstva bude CNN
udelame to jako v efficient DET
budeme tam mit 2 nekolik convolutions
