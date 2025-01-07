## PointNet: *Deep Learning on Point Sets for 3D Classification and Segmentation*

### Introduction

### Citation

### Installation

### Usage
To train a model to classify point clouds sampled from 2D shapes:

    python train.py --data **
    # python train.py

Log files and network parameters will be saved to `log` folder in default. 

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir log

After the above training, we can evaluate the model and output some visualizations of the error cases.

    python evaluate.py --visu

If you'd like to prepare your own data, you can refer to some helper functions in `utils/data_prep_util.py` for saving and loading txt files.
# PCDProcessor
