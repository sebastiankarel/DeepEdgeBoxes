# DeepEdgeBoxes: A Deep Learning Approach to Object Localisation on Edge Images

This implementation serves as proof of concept for applying a convolutional neural network on local edge image patches to generate object proposals with an exhaustive sliding window.

## Training
To generate the training, validation and test datasets, run `pvoc07_util.py` after specifying the input and output paths in `pvoc07_paths.txt`. To run the training, first specify the location of the dataset in the `train_configs.txt`, as created by `pvoc07_util.py`.  Then simply run `Train.py` with the following optional arguments: `edge_type`, `batch_size` and `epochs`. `edge_type` may be one of the following `single_canny`, `multi_canny` `rgb_canny` or `hed`, so e.g. `edge_type=single_canny`. Note that the annotations must be in the format that is provided by the Pascal VOC 2007 Dataset.

## Evaluation
Similar to training, to run the evaluation, set the test set path in `eval_configs.txt` and run `Eval.py`, again using the `edge_type` argument. This will evaluate the whole dataset and generate a log file with the recall for IoU=0.3, IoU=0.4, IoU=0.5, IoU=0.6 and IoU=0.7, as well as 10, 100 and 1000 proposals. For a visual inspection, simply run `EvalVisual.py`.
